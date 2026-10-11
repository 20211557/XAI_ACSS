"""공정성 검증 및 Bias Mitigation 모듈 (보호 속성: 성별, 연령대).

- metrics.py    : Disparate Impact Ratio (0.8 ~ 1.25 벗어나면 WARNING),
                  Equalized Odds (TPR/FPR 차이)
- mitigation.py : Reweighting, Threshold Adjustment 및 적용 전/후 비교
"""
