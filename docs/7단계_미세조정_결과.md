# 7단계 미세 조정: 지원님 자료로 대체 (2026-11-02)

지원님 모델 가중치(/home/jiwon934/eg2-lab/models/eg2-classroom/model.safetensors)는 지원님만 읽을 수 있어(권한 600) 복사하지 못했다.
그래서 지원님이 공유한 학습 데이터와 스크립트(/var/tmp/eg2-share-jiwon934/step7)를 그대로 써서 다시 학습했다.
- 데이터: train 303쌍(101문서), eval 60쌍(20문서)
- 설정: train_ft.py 그대로. 3 epoch, batch 8 x accum 2, lr 2e-5, bf16, MatryoshkaLoss

결과 (평가셋: 지원님 eval.jsonl, eval_ir.py)
| 모델 | accuracy@1 | accuracy@3 | ndcg@10 | 학습 시간 |
| --- | --- | --- | --- | --- |
| 기준 google/embeddinggemma-2 | 0.833 | 0.967 | 0.930 | - |
| 다시 학습 (step7_jiwon/models/eg2-classroom) | 0.850 | 0.967 | 0.934 | 250.6초 (GPU) |
| 지원님 결과 (10-23, step7_result.md) | 0.850 | 0.967 | 0.934 | 236.5초 |
| 기존 ~/eg2-lab/models/eg2-classroom (10-26 생성) | 0.867 | 0.967 | 0.940 | - |

- 다시 학습한 결과가 지원님 숫자와 똑같다(epoch별 acc@1도 0.850 → 0.867 → 0.850로 같음). 그래서 지원님 모델을 대체해 쓸 수 있다.
- 기존 models/eg2-classroom은 10-26에 만들어진 모델로, 같은 평가셋에서 조금 더 높다(지원님 2회차 최고값과 같음). 덮어쓰지 않았다.

파일
- ~/eg2-lab/step7_jiwon/: data/, make_data.py, train_ft.py, eval_ir.py, train.log(지원님), train_mine.log, eval_base.log, eval_ft.log
- ~/eg2-lab/step7_jiwon/models/eg2-classroom/: 다시 학습한 모델 (548MB)
