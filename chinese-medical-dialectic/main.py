from utils import call_large_model, score, RuleIndex


import csv


def detect_combo(symptom: str) -> bool:
    """
    检测是否属于“痰湿内蕴兼气虚发热”复合证型：
    条件：
      痰湿相关 + 气虚相关 + 发热/内热/口干口苦 共同出现
    """
    cond_phlegm = any(k in symptom for k in [
        "痰涎量多", "痰涎量中等", "痰涎量", "痰多", "唾液粘稠", "口中发黏"
    ])
    cond_qixu = any(k in symptom for k in [
        "劳时即乏", "劳时则气短", "动时即乏", "不动即乏", "乏力", "气短", "精神疲惫"
    ])
    cond_heat = any(k in symptom for k in [
        "自觉发热", "发热", "手足心发热", "口干口苦"
    ])
    return cond_phlegm and cond_qixu and cond_heat


def predict(symptom):
    """
    :param symptom: str, 症状描述
    :return zx_predict, zf_predict：str, 依次为症型，治法描述
    """
    # 1. 专门处理“痰湿内蕴兼气虚发热”复合证型
    if detect_combo(symptom):
        return "痰湿内蕴兼气虚发热", "健脾燥湿，化痰利浊"

    # 2. 数据驱动（Naive Bayes）
    if rule_index is not None:
        lab, conf = rule_index.predict(symptom)
        # 若预测为基础“痰湿内蕴”但包含复合征热与气虚迹象，提升为复合证型
        if lab is not None:
            zx, zf = lab
            if zx == "痰湿内蕴" and detect_combo(symptom):
                return "痰湿内蕴兼气虚发热", zf
            if conf >= 0.35:
                return zx, zf

    # 3. 静态关键字规则
    symptoms = rule_index.split_symptoms(symptom)
    for s in symptoms:
        for keyword, result in rule_index.static_rules.items():
            if keyword in s:
                return result["证型"], result["治法"]

    # 4. 模型兜底
    model_result = call_large_model(symptom)
    return model_result['证型'], model_result['治法']


train_data = './datasets/68f201a04e0f8ad44a62069b-momodel/train_data.csv'
symptoms_data = []
ZX = []
ZF = []
samples = []
with open(train_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        text = row[1]
        zx = row[2]
        zf = row[3]
        symptoms_data.append(text)   # 提取症状列
        ZX.append(zx)                # 提取证型列
        ZF.append(zf)                # 提取治法列
        samples.append((text, zx, zf))

rule_index = RuleIndex()
rule_index.fit(samples)

predict_zx = []
predict_zf = []
for symptom in symptoms_data:
    zx, zf = predict(symptom)
    predict_zx.append(zx)
    predict_zf.append(zf)

final_score = score(predict_zx, predict_zf, ZX, ZF)
print(final_score)