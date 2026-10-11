"""회귀 계수(β) 표와 변수 간 WoE 상관을 출력한다. 부호 검사는 끄고 보여 준다.

    python -m training.scorecard.report_fit
"""
from training.scorecard.binning import DEFAULT_KINDS, apply_woe, fit_woe_tables
from training.scorecard.features import build_training_frame
from training.scorecard.fit import fit_logit
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.split import stratified_split


def main() -> None:
    train, _ = stratified_split(build_training_frame(load_uci()), LABEL, 0.3, 42)
    woe = apply_woe(train, fit_woe_tables(train, DEFAULT_KINDS))
    result = fit_logit(woe, train[LABEL], check_signs=False)

    print(f"절편 β0 = {result.intercept:+.4f}\n")
    print("| 변수 | β | 표준오차 | 부호 |")
    print("|---|---:|---:|---|")
    for name, beta in result.betas.items():
        print(f"| {name} | {beta:+.4f} | {result.bse[name]:.4f} | {'정상(음수)' if beta < 0 else '**양수**'} |")

    print("\n변수 간 WoE 상관 (학습 데이터)")
    print(woe.corr().round(3).to_string())


if __name__ == "__main__":
    main()
