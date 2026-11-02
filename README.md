# embedding_gemma2 — Jetson에서 EmbeddingGemma 2로 텍스트·이미지·영상·음성 검색과 카메라 응용

AIGJ_01_013 이진행

Jetson Orin Nano에 EmbeddingGemma 2를 올려 텍스트·이미지·영상·음성을 하나의 벡터 공간에서 검색하고, 이를 카메라 응용(물체 검색, 동작·행동 인식, 카메라 2대 접근 감지)으로 넓힌 실습입니다. 1~6단계는 공통 기반, 7단계는 텍스트 검색 미세 조정, 8~10단계는 카메라 응용입니다.

- **최종 보고서:** [report/최종보고서.md](report/최종보고서.md)
- **10단계 결과 모음:** [docs/step10/INDEX.md](docs/step10/INDEX.md)

## 핵심 결과

| 단계 | 결과 |
|---|---|
| 1~2. 환경 구축 | torch 2.14.1+cu132, GPU 행렬 곱 확인, 임베딩 (768,) |
| 3. 텍스트 검색 | 768/256/128차원 모두 정답 1위, 질의 2개 약 165ms |
| 4. 이미지·영상 검색 | 이미지 질의 4개 중 3개 정답, 영상은 1fps 프레임 평균으로 대체 |
| 5. 음성 검색 | 실제 녹음으로 질의 3개 모두 정답 |
| 6. 성능 측정 | 텍스트 146ms, 이미지 5장 2.9초, 전체 3.4초, 최대 RAM 약 6.1GB |
| 7. 도메인 미세 조정 | acc@1 0.783 → 0.800(v2), 일반 질의 성능 유지 |
| 8. 물체 추출·검색 | 강의실에서는 YOLOE가 가장 적합 |
| 9. 동작·행동 인식 | 측면 만세·박수 규칙 개선, 행동 4종 중 3종 성공 |
| 10. 카메라 2대 접근 감지 | fused 거리 오차 -10cm, 충돌 알람 4/5, 헛알람 0 |

## 폴더 구성

```
report/        최종 보고서 (Markdown, 그림)
docs/          단계별 결과 기록 (6·7·8·9단계, 10단계 안내)
  step10/      10단계 측정 표·로그(csv), 시나리오·이동 타겟(10-8) 결과, 공개 데이터(EPFL) 결과
src/           단계별 최종 코드
  archive/     중간 버전 (7단계 v1, 9단계 pose_app v1~v6·train_pose v1~v3, 10단계 EPFL 실험)
ipcam.example  IP 카메라 접속 정보 견본
```

## 코드와 실행 순서

모든 스크립트는 젯슨의 `~/eg2-lab/`에서 실행했습니다. 실행 전 가상환경을 켭니다: `source ~/eg2/bin/activate`

| 순서 | 파일 | 단계 | 하는 일 |
|---|---|---|---|
| 1 | `step01_02_setup/check.py`, `eg2.py` | 1~2 | GPU 확인, 모델 로드와 공통 `load()`·`encode()` |
| 2 | `step03_text/test_text.py` 외 | 3 | 한국어 질의 → 문서 검색, 차원(768/256/128)·프롬프트 비교 |
| 3 | `step04_vision/capture.py`, `test_vision.py`, `test_video_frames.py` | 4 | C920 캡처, 이미지 검색, 영상 프레임 평균 검색 |
| 4 | `step05_audio/test_audio.py` | 5 | 16kHz WAV 음성 검색 |
| 5 | `step06_bench/bench.py` | 6 | 구성별 RAM·지연·GPU 사용률 측정 |
| 6 | `step07_finetune/make_data.py`, `train_ft_v2.py`, `eval_ir.py`, `test_text_ft2.py` | 7 | 강의실 질의-문서 쌍 생성, 미세 조정(v2: 1에폭, lr 1e-5), 평가 |
| 7 | `step08_object_search/app.py` | 8 | YOLO/YOLOE로 물체 추출 → EmbeddingGemma 2 글 검색 웹앱 (:8002) |
| 8 | `step09_pose_behavior/pose_app_v4_side.py` | 9 | 관절 기반 동작 판정, `--behavior` 행동 패턴 (최종판) |
| 9 | `step09_pose_behavior/train_pose_v4.py`, `pose_eval_v4.py`, `eval_gru.py` | 9 | GRU 동작 모델 학습·평가 (실시간 앱에서는 끔) |
| 10 | `step10_proximity/calib_intrinsics.py` → `calib_extrinsics.py` | 10 | 카메라별 렌즈 보정, 바닥 기준점으로 두 카메라 외부 보정 |
| 11 | `step10_proximity/proximity_app_leave.py` | 10 | 타겟 찾기 + 어깨 삼각측량 + 접근/이탈 알람 (이탈 알람 고친 최종판) |

10단계 카메라 배치와 실행 방법은 [docs/10단계_실습안내.md](docs/10단계_실습안내.md)에 있습니다.

## 환경

- Jetson Orin Nano (8GB), L4T R39.2.1(JetPack 7 계열), Ubuntu 24.04, Python 3.12, NVMe 부팅
- torch 2.14.1+cu132, sentence-transformers 6.2.0, 모델 google/embeddinggemma-2 (bfloat16), ultralytics(YOLO11n, YOLO11n-pose, YOLOE-11s)
- 카메라: Logitech C920 2대, IP 카메라 TP-Link VIGI

## 올리지 않은 것

- 모델 가중치와 미세 조정 체크포인트(`models/`, `runs/eg2-classroom*`, `*.pt`, `*.safetensors`, `*.ts`): 용량
- 영상·음성·이미지 원본(`media/`), 동작 특징 데이터(`data/`), 실행 로그(`*.log`): 용량, 사람이 찍힌 자료
- 7단계 학습·평가 데이터(jsonl): 팀원(정지원) 공유 자료라 넣지 않음, `make_data.py`로 다시 만들 수 있음
- 접속 정보(`~/.ipcam`, `.cam.env`): 형식은 `ipcam.example` 참고
- 8단계 결과 기록(`docs/8단계_결과기록_팀원.md`), 9단계 정리, 10단계 실제 교실 결과는 팀원과 함께 만든 기록이며 출처를 각 문서에 적었습니다.
