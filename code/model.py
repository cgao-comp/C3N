import torch
import torch.nn as nn
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
import torch.nn.functional as F
import clip

C3N_param = {
    'fc1_in': 512,
    'fc1_out': 256,
    'fc2_out': 128,
    'fc3_out': 64,
    'consis_fc1_out': 128,
    'consis_fc2_out': 64,
    'fusion_fc1_out': 64,
    'fusion_fc2_out': 32
}

class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, layer_num, head, tgt_seq, src_seq):
        super(TransformerEncoder, self).__init__()
        self.layer_num = layer_num
        self.multihead_attns_t = nn.ModuleList([nn.MultiheadAttention(model_dim, head) for _ in range(self.layer_num)])
        self.multihead_attns_s = nn.ModuleList([nn.MultiheadAttention(model_dim, head) for _ in range(self.layer_num)])
        self.LN_ts1 = nn.ModuleList([nn.LayerNorm([tgt_seq, model_dim]) for _ in range(self.layer_num)])
        self.LN_ss1 = nn.ModuleList([nn.LayerNorm([src_seq, model_dim]) for _ in range(self.layer_num)])
        self.LN_ts2 = nn.ModuleList([nn.LayerNorm([tgt_seq, model_dim]) for _ in range(self.layer_num)])
        self.LN_ss2 = nn.ModuleList([nn.LayerNorm([src_seq, model_dim]) for _ in range(self.layer_num)])
        FF_K = 4
        self.ff_ts = nn.ModuleList([nn.Sequential(nn.Linear(model_dim, FF_K * model_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(FF_K * model_dim, model_dim)) for _ in range(self.layer_num)])
        self.ff_ss = nn.ModuleList([nn.Sequential(nn.Linear(model_dim, FF_K * model_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(FF_K * model_dim, model_dim)) for _ in range(self.layer_num)])
    def forward(self, tgt, src):
        for multihead_attn_t, multihead_attn_s, ff_t, ff_s, LN_t1, LN_s1, LN_t2, LN_s2 in zip(self.multihead_attns_t, self.multihead_attns_s, self.ff_ts, self.ff_ss, self.LN_ts1, self.LN_ss1, self.LN_ts2, self.LN_ss2):
            res_t = tgt 
            res_s = src
            tgt = tgt.permute(1, 0, 2)  
            src = src.permute(1, 0, 2)  
            tgt, _ = multihead_attn_t(tgt, src, src)
            tgt = tgt.permute(1, 0, 2)  
            tgt = LN_t1(tgt + res_t)
            src, _ = multihead_attn_s(src, tgt, tgt)
            res_t = tgt
            tgt = ff_t(tgt)
            tgt = LN_t2(tgt + res_t)
            src = src.permute(1, 0, 2) 
            src = LN_s1(src + res_s)
            res_s = src
            src = ff_s(src)
            src = LN_s2(src + res_s)

        return tgt, src


class C3N(nn.Module):
    def __init__(self, args):
        super(C3N, self).__init__()
        self.is_weibo = False
        if args.dataset == 'weibo':
            clip_model, _ = load_from_name(args.clip_model, device=args.device, download_root='./')
            self.is_weibo = True
            self.fc_768 = nn.Linear(768, 512)
        else:
            clip_model, _ = clip.load('ViT-B/16', args.device)
        self.clip_model = clip_model.float()
        self.finetune = args.finetune
        Ks = [1, 2, 3]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=args.conv_out, kernel_size=(K, args.crop_num)) for K in Ks])
        self.convs_ob = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=args.conv_out, kernel_size=(K, args.word_num)) for K in Ks])
        self.device = args.device
        self.logit_scale = 100
        Conv_out = len(Ks) * args.conv_out
        self.transformer = TransformerEncoder(model_dim=512, layer_num=args.layer_num, head=8, tgt_seq=args.word_num, src_seq=args.crop_num)
        self.fc_st1 = nn.Sequential(nn.Linear(C3N_param['fc1_in'], C3N_param['fc1_out']),
                                    nn.ReLU(),
                                    nn.Linear(C3N_param['fc1_out'], C3N_param['fc2_out']),
                                    nn.ReLU(),
                                    nn.Linear(C3N_param['fc2_out'], C3N_param['fc3_out']),
                                    nn.ReLU())
        self.fc_consis1 = nn.Sequential(nn.Linear(Conv_out, C3N_param['consis_fc1_out']),
                                        nn.ReLU(),
                                        nn.Linear(C3N_param['consis_fc1_out'], C3N_param['consis_fc2_out']),
                                        nn.ReLU())
        self.fc_ob1 = nn.Sequential(nn.Linear(C3N_param['fc1_in'], C3N_param['fc1_out']),
                                    nn.ReLU(),
                                    nn.Linear(C3N_param['fc1_out'], C3N_param['fc2_out']),
                                    nn.ReLU(),
                                    nn.Linear(C3N_param['fc2_out'], C3N_param['fc3_out']),
                                    nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(C3N_param['consis_fc2_out'] + C3N_param['fc3_out'] * 2, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU())
        self.fc = nn.Linear(C3N_param['fusion_fc2_out'], out_features=2)
        self.dropout = nn.Dropout(args.dropout_p)

        if not args.finetune:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def calculate_similarity(self, txt_fea, img_fea):
        txt_fea_ = txt_fea / txt_fea.norm(dim=-1, keepdim=True)
        img_fea_ = img_fea / img_fea.norm(dim=-1, keepdim=True)
        img_fea_T = torch.transpose(img_fea_, 1, 2) 
        sim = torch.matmul(txt_fea_, img_fea_T) 
        sim_T = torch.transpose(sim, 1, 2)
        sim = (self.logit_scale * sim).unsqueeze(1)
        sim_T = (self.logit_scale * sim_T).unsqueeze(1)

        fea_maps = [F.relu(conv(sim)).squeeze(3) for conv in self.convs]
        consis_fea_avg = [F.avg_pool1d(i, i.shape[2]).squeeze(2) for i in fea_maps]
        consis_fea_avg = torch.cat(consis_fea_avg, 1)

        st_embed_pos = torch.mean(txt_fea_, 1)
        ob_embed_pos = torch.mean(img_fea_, 1)

        return st_embed_pos, ob_embed_pos, consis_fea_avg

    def forward(self, data):

        word_features = data['word_features'].to(self.device)
        crop_features = data['crop_features'].to(self.device)
        if self.finetune:
            word_features = self.clip_model.my_encode_text(word_features)
            if self.is_weibo:
                word_features = self.fc_768(word_features)
            img_num = crop_features.shape[1]
            crop_features = crop_features.reshape(-1, crop_features.shape[2], crop_features.shape[3], crop_features.shape[4])
            crop_features = self.clip_model.encode_image(crop_features)
            crop_features = crop_features.reshape(-1, img_num, crop_features.shape[1])
        wi_fea, iw_fea = self.transformer(word_features, crop_features)
        st, ob, consis = self.calculate_similarity(wi_fea, iw_fea)
        st = self.dropout(self.fc_st1(st))
        ob = self.dropout(self.fc_ob1(ob))
        consis = self.dropout(self.fc_consis1(consis))
        combined_features = torch.cat([st, ob, consis], dim=-1)
        fused = self.dropout(self.fusion(combined_features))
        x = self.fc(fused)
        logit = F.log_softmax(x, dim=-1)

        return logit

