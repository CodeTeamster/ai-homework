import re
import math
from collections import defaultdict, Counter


STOP_HINTS = [
    "患者", "身高", "体重", "西医诊断", "分期", "既往史", "主诉", "体格检查", "备注",
    "大小约", "cm", "mm", "nm", "mn", "m", "x", "*", ":", "；", ";"
]
CANON_MAP = {
    "痰涎量多": "痰多",
    "痰涎量中等": "痰多",
    "唾液粘稠": "痰稠",
    "口中发黏": "口黏",
    "劳时即乏": "乏力",
    "动时即乏": "乏力",
    "不动即乏": "乏力",
    "劳时则气短": "气短",
    "自觉发热": "发热",
    "睡眠时常常觉醒或睡而不稳，晨醒过早": "睡眠不稳",
    "睡眠不足4小时": "睡眠不足",
    "患者形体肥胖": "肥胖",
    "形体肥胖": "肥胖",
    "淋巴结：肿大": "淋巴结肿大",
    "颈部淋巴结：肿大": "淋巴结肿大",
    "腋下淋巴结：肿大": "淋巴结肿大",
    "腹股沟淋巴结：肿大": "淋巴结肿大",
    "脾脏情况：脾大": "脾大",
    "脾稍大": "脾大",
    "脾大，脾脏": "脾大",
    "纳差": "食欲差",
    "不欲食，尚能进食，食欲稍减": "食欲差",
    "纳食时有减少，经常呕恶": "食欲差",
}
STATIC_RULES = {
    "痰涎量多": {"证型": "痰湿内蕴", "治法": "健脾燥湿，化痰利浊"},
    "舌暗红": {"证型": "痰湿内蕴", "治法": "健脾燥湿，化痰利浊"},
    "劳时即乏": {"证型": "气阴两虚", "治法": "益气养阴，生津润燥"},
    "腰膝酸软": {"证型": "气阴两虚", "治法": "益气养阴，生津润燥"},
    "形体肥胖": {"证型": "脾虚痰湿", "治法": "健脾益气，化湿祛痰"},
    "舌瘦削": {"证型": "痰瘀互结", "治法": "豁痰祛瘀，软坚散结"},
    # 扩展
    "淋巴结肿大": {"证型": "脾虚痰湿", "治法": "健脾益气，化湿祛痰"},
    "脾大": {"证型": "脾虚痰湿", "治法": "健脾益气，化湿祛痰"},
    "痰多": {"证型": "痰湿内蕴", "治法": "健脾燥湿，化痰利浊"},
    "肥胖": {"证型": "脾虚痰湿", "治法": "健脾益气，化湿祛痰"},
    "乏力": {"证型": "气阴两虚", "治法": "益气养阴，生津润燥"},
    # 发热 + 痰湿 + 气虚迹象可能属于复合证型
    "自觉发热": {"证型": "痰湿内蕴兼气虚发热", "治法": "健脾燥湿，化痰利浊"},
    "手足心发热": {"证型": "痰湿内蕴兼气虚发热", "治法": "健脾燥湿，化痰利浊"},
    "口干口苦": {"证型": "痰湿内蕴兼气虚发热", "治法": "健脾燥湿，化痰利浊"},
}
DELIMS = re.compile(r"[，。、；;,\s\n]+")


class RuleIndex:
    def __init__(self):
        self.vocab = set()
        # list of (证型, 治法)
        self.labels = []
        self.label_to_idx = {}
        self.label_doc_counts = Counter()
        # 以 token 为键值，记录所有 samples 下 token 在多少 samples 中出现过
        self.token_df = Counter()
        # 以(证型, 治法)为键值，记录所有 samples 下 labels 为(证型, 治法)对应 token 出现次数
        self.token_label_counts = defaultdict(lambda: Counter())
        self.sample_num = 0
        self.static_rules = STATIC_RULES

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        t = re.sub(r"[0-9A-Za-z\.\-]+", "<NUM>", text)
        for k, v in CANON_MAP.items():
            t = t.replace(k, v)
        return t

    def _tokenize(self, text: str):
        t = self._normalize_text(text)
        parts = [p.strip() for p in DELIMS.split(t) if p.strip()]
        tokens = []
        for p in parts:
            # 过滤无信息 token
            if any(h in p for h in STOP_HINTS):
                continue
            # 单字常见无信息过滤（保留舌/苔常见颜色单字）
            if len(p) == 1 and p not in ("红", "白", "黄", "润", "燥"):
                continue
            tokens.append(p)
        return tokens

    def split_symptoms(self, symptom):
        delimiters = r"[、，。；；\s]"
        return [s.strip() for s in re.split(delimiters, symptom) if s.strip()]

    def fit(self, samples):
        # samples: list[(text, zx, zf)]
        self.sample_num = len(samples)
        label_set = []
        for _, zx, zf in samples:
            label_set.append((zx.strip(), zf.strip()))
        self.labels = sorted(set(label_set))
        self.label_to_idx = {lab: i for i, lab in enumerate(self.labels)}

        # 文档级二值计数
        for text, zx, zf in samples:
            lab = (zx.strip(), zf.strip())
            self.label_doc_counts[lab] += 1
            toks = set(self._tokenize(text))
            for tok in toks:
                self.token_df[tok] += 1
                self.token_label_counts[lab][tok] += 1

        # 构建词表：去除过普遍或过稀少的 token
        self.vocab = {
            tok for tok, df in self.token_df.items()
            if 1 <= df <= int(0.8 * self.sample_num)  # 出现频率上限 80%
        }

    def predict(self, text, alpha=1.0):
        toks = self._tokenize(text)
        toks = [t for t in toks if t in self.vocab]
        if not toks:
            return None, 0.0

        vocab_size = max(1, len(self.vocab))
        # 先验
        scores = []
        for lab in self.labels:
            # 朴素贝叶斯加一平滑先验 P(label)
            prior = (self.label_doc_counts[lab] + 1.0) / (self.sample_num + len(self.labels))
            s = math.log(prior)
            lab_count = self.label_doc_counts[lab]
            for t in toks:
                ct = self.token_label_counts[lab].get(t, 0)
                # 加法平滑条件概率 P(token|label)
                s += math.log((ct + alpha) / (lab_count + alpha * vocab_size))
            scores.append((s, lab))

        scores.sort(reverse=True, key=lambda x: x[0])
        top_s, top_lab = scores[0]
        second_s = scores[1][0] if len(scores) > 1 else -1e9

        # 简单置信度：分差的 Sigmoid，叠加匹配比例
        margin = max(0.0, top_s - second_s)
        # [0, 0.5)
        conf_margin = 1.0 / (1.0 + math.exp(-margin)) - 0.5
        # 匹配越多越好，6个及以上饱和
        match_ratio = min(1.0, len(toks) / 6.0)
        confidence = min(1.0, 0.6 * match_ratio + 0.8 * conf_margin)

        return top_lab, confidence