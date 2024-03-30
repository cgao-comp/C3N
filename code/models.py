import torch
import torch.nn as nn
from cn_clip.clip import load_from_name
import torch.nn.functional as F
import clip

CNNSim2_param = {
    'st_fc1_in': 512,
    'st_fc1_out': 256,
    'st_fc2_out': 128,
    'st_fc3_out': 64,
    'consis_fc1_out': 128,
    'consis_fc2_out': 64,
    'fusion_fc1_out': 64,
    'fusion_fc2_out': 32
}


class TransformerEncoder(nn.Module):
    """ general attention for tgt & src from different modality
    """

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
            res_t = tgt  # [B, Seq, D]
            res_s = src
            tgt = tgt.permute(1, 0, 2)  # [Seq, B, D]
            src = src.permute(1, 0, 2)  # [Seq, B, D]
            tgt_new, _ = multihead_attn_t(tgt, src, src)
            src_new, _ = multihead_attn_s(src, tgt, tgt)
            tgt_new = tgt_new.permute(1, 0, 2)  # [B, Seq, D]
            src_new = src_new.permute(1, 0, 2)  # [B, Seq, D]
            tgt_new = LN_t1(tgt_new + res_t)
            src_new = LN_s1(src_new + res_s)
            res_t = tgt_new
            res_s = src_new
            tgt_new = ff_t(tgt_new)
            src_new = ff_s(src_new)
            tgt = LN_t2(tgt_new + res_t)
            src = LN_s2(src_new + res_s)

        return tgt, src


class C3N(nn.Module):
    def __init__(self, args):
        super(C3N, self).__init__()
        self.is_weibo = False
        if args.dataset == 'weibo':
            clip_model, _ = load_from_name('ViT-B-16', device=args.device, download_root='/sda/qiaojiao/pretrained_models/cn-clip/')
            self.is_weibo = True
            self.fc_768 = nn.Linear(768, 512)
            self.clip_model = clip_model.float()

        else:
            clip_model, _ = clip.load('ViT-B/16', args.device, download_root="/sda/qiaojiao/pretrained_models/clip/")

        self.finetune = args.finetune
        if args.dataset == "twitter":
            Ks_word = args.conv_kernel
        else:
            Ks_word = args.conv_kernel
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=args.conv_out, kernel_size=(K, args.crop_num)) for K in Ks_word])
        self.device = args.device
        self.logit_scale = 100
        Conv_out = len(Ks_word) * args.conv_out
        self.transformer = TransformerEncoder(model_dim=512, layer_num=args.layer_num, head=8, tgt_seq=args.st_num, src_seq=args.crop_num)
        self.fc_st1 = nn.Sequential(nn.Linear(CNNSim2_param['st_fc1_in'], CNNSim2_param['st_fc1_out']),
                                    nn.ReLU(),
                                    nn.Linear(CNNSim2_param['st_fc1_out'], CNNSim2_param['st_fc2_out']),
                                    nn.ReLU(),
                                    nn.Linear(CNNSim2_param['st_fc2_out'], CNNSim2_param['st_fc3_out']),
                                    nn.ReLU())
        self.fc_consis1 = nn.Sequential(nn.Linear(Conv_out, CNNSim2_param['consis_fc1_out']),
                                        nn.ReLU(),
                                        nn.Linear(CNNSim2_param['consis_fc1_out'], CNNSim2_param['consis_fc2_out']),
                                        nn.ReLU())
        self.fc_ob1 = nn.Sequential(nn.Linear(CNNSim2_param['st_fc1_in'], CNNSim2_param['st_fc1_out']),
                                    nn.ReLU(),
                                    nn.Linear(CNNSim2_param['st_fc1_out'], CNNSim2_param['st_fc2_out']),
                                    nn.ReLU(),
                                    nn.Linear(CNNSim2_param['st_fc2_out'], CNNSim2_param['st_fc3_out']),
                                    nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(CNNSim2_param['consis_fc2_out'] + CNNSim2_param['st_fc3_out'] * 2, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU())
        self.fc = nn.Linear(CNNSim2_param['fusion_fc2_out'], out_features=2)
        self.dropout = nn.Dropout(args.dropout_p)

        if not args.finetune:
                
            for name, param in self.clip_model.named_parameters():
                if name.find("logit_scale") != -1:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                
                                    
    def similarity_weight(self, txt_fea, img_fea):
        """ compute the clip_sim to return batch features
        """
        txt_fea_ = txt_fea / txt_fea.norm(dim=-1, keepdim=True)
        img_fea_ = img_fea / img_fea.norm(dim=-1, keepdim=True)
        img_fea_T = torch.transpose(img_fea_, 1, 2)  # (8,512,5)
        sim = torch.matmul(txt_fea_, img_fea_T)  # (8,15,5)
        sim = (self.logit_scale * sim).unsqueeze(1)

        fea_maps = [F.relu(conv(sim)).squeeze(3) for conv in self.convs] # !
        consis_fea_avg = [F.avg_pool1d(i, i.shape[2]).squeeze(2) for i in fea_maps]
        consis_fea_avg = torch.cat(consis_fea_avg, 1)

        st_embed_pos = txt_fea[:, 0, :]
        ob_embed_pos = img_fea[:, 0, :]

        return st_embed_pos, ob_embed_pos, consis_fea_avg
    
    def clip_encode(self, text_input, crop_input, n_word_input):
        sentence_features = self.clip_model.encode_text(text_input) # (B, 512)
        n_num = n_word_input.shape[1]
        n_word_input = n_word_input.reshape(-1, n_word_input.shape[2])
        word_features = self.clip_model.encode_text(n_word_input)
        word_features = word_features.reshape(-1, n_num, word_features.shape[1])
        word_features = torch.cat([sentence_features.unsqueeze(1), word_features], dim=1)
        
        img_num = crop_input.shape[1] 
        crop_input = crop_input.reshape(-1, crop_input.shape[2], crop_input.shape[3], crop_input.shape[4])
        crop_features = self.clip_model.encode_image(crop_input)
        crop_features = crop_features.reshape(-1, img_num, crop_features.shape[1])
        
        return word_features, crop_features
    

    def forward(self, data):
        text_input = data['text_input']
        crop_input = data['crop_input']
        if self.is_weibo:
            word_features, crop_features = self.clip_encode(text_input, crop_input, data['n_word_input'])
        else:
            word_features = torch.cat([text_input.unsqueeze(1), data['n_word_input']], dim=1)
            crop_features = crop_input
        
        wi_fea, iw_fea = self.transformer(word_features, crop_features)
        st, ob, consis = self.similarity_weight(wi_fea, iw_fea)
        st = self.dropout(self.fc_st1(st))
        ob = self.dropout(self.fc_ob1(ob))
        consis = self.dropout(self.fc_consis1(consis))
        combined_features = torch.cat([st, ob, consis], dim=-1)
        fused = self.dropout(self.fusion(combined_features))
        x = self.fc(fused)
        logit = F.log_softmax(x, dim=-1)

        return logit
