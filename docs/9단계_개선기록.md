# 9단계 개선 (2026-10-26)

웹캠 대신 인터넷 샘플 영상(media/pose/SOURCES.txt)으로 측정. 원본 파일(pose_app.py, train_pose.py, models/action.pt, step9_result.md)은 그대로 두고 새 파일로만 작업했다.

## 1. 9-6 GRU 동작 모델 다시 학습

### 바꾼 것
- 학습 데이터: 정지원님 웹캠 데이터(data/action_jiwon: arms_crossed·clap·idle, 한 사람·한 각도)에 샘플 영상 특징을 더함 (extract_feats.py, 15fps로 맞춤)
  - v2: + 팔짱 2개, 박수 1개, 손 흔들기 1개 영상
  - v3: v2 + 박수 4개(Mixkit 사무실·세미나·회의·집단), 기본 5개(걷는 사람·서 있는 사람)
  - v3m: v3 + 좌우 뒤집기 증강(train_pose_v2.py --mirror, 학습 윈도 2배)
  - v4m: v3m + 손 흔들기 5개 영상
- 평가는 학습에 쓰지 않은 영상만: 팔짱 mk29963, 박수 applause_wikiconvention, 손 흔들기 wave_example, 기본 vtest.avi (data/action_test, eval_gru.py)

### 테스트 영상 결과 (윈도 30프레임, stride 5)
| 모델 | 팔짱 | 박수 | 기본 | 손 흔들기 | 전체 정확도 | 평균 최고 확률 |
| --- | --- | --- | --- | --- | --- | --- |
| action.pt (원래) | 100% | 0% | 0% | 0% | 34% | 0.30~0.50 |
| action_v2.pt | 100% | 0% | 0% | 63% | 50% | 0.45~0.63 |
| action_v3.pt | 90% | 31% | 79% | 20% | 62% | 0.82~0.91 |
| **action_v3m.pt** | **100%** | **100%** | **76%** | 0% | **69%** | 0.82~0.98 |
| action_v4m.pt | 100% | 50% | 82% | 0% | 64% | 0.81~0.96 |

원래 모델은 확률 0.7을 넘는 판정이 하나도 없었는데(그래서 앞의 표에서 GRU가 전부 0), v3m은 테스트 윈도 122개 중 112개가 0.7을 넘고 그중 81개가 맞았다.

<details><summary>eval_gru.py 원본 출력</summary>

### models/action.pt (라벨 ['arms_crossed', 'clap', 'idle', 'wave'], window 30)
| arms_crossed | 42 | 42 | 100% | 0.45 | {'arms_crossed': 42} | 0(0) | 0(0) |
| clap | 16 | 0 | 0% | 0.46 | {'arms_crossed': 16} | 0(0) | 4(0) |
| idle | 34 | 0 | 0% | 0.30 | {'clap': 34} | 0(0) | 0(0) |
| wave | 30 | 0 | 0% | 0.50 | {'arms_crossed': 30} | 0(0) | 14(0) |
| 전체 | 122 | 42 | 34% | | | | |
### models/action_v2.pt (라벨 ['arms_crossed', 'clap', 'idle', 'wave'], window 30)
| arms_crossed | 42 | 42 | 100% | 0.56 | {'arms_crossed': 42} | 0(0) | 42(42) |
| clap | 16 | 0 | 0% | 0.63 | {'wave': 12, 'arms_crossed': 3, 'idle': 1} | 6(0) | 10(0) |
| idle | 34 | 0 | 0% | 0.49 | {'arms_crossed': 29, 'wave': 4, 'clap': 1} | 0(0) | 13(0) |
| wave | 30 | 19 | 63% | 0.45 | {'wave': 19, 'idle': 11} | 0(0) | 5(5) |
| 전체 | 122 | 61 | 50% | | | | |
### models/action_v3m.pt (라벨 ['arms_crossed', 'clap', 'idle', 'wave'], window 30)
| arms_crossed | 42 | 42 | 100% | 0.95 | {'arms_crossed': 42} | 42(42) | 42(42) |
| clap | 16 | 16 | 100% | 0.90 | {'clap': 16} | 15(15) | 16(16) |
| idle | 34 | 26 | 76% | 0.82 | {'idle': 26, 'clap': 6, 'wave': 2} | 25(24) | 28(26) |
| wave | 30 | 0 | 0% | 0.98 | {'clap': 30} | 30(0) | 30(0) |
| 전체 | 122 | 84 | 69% | | | | |
### models/action_v4m.pt (라벨 ['arms_crossed', 'clap', 'idle', 'wave'], window 30)
| arms_crossed | 42 | 42 | 100% | 0.81 | {'arms_crossed': 42} | 42(42) | 42(42) |
| clap | 16 | 8 | 50% | 0.91 | {'wave': 8, 'clap': 8} | 16(8) | 16(8) |
| idle | 34 | 28 | 82% | 0.86 | {'idle': 28, 'clap': 6} | 25(25) | 31(27) |
| wave | 30 | 0 | 0% | 0.96 | {'clap': 30} | 30(0) | 30(0) |
| 전체 | 122 | 78 | 64% | | | | |
</details>

### 남은 한계 (GRU)
- 손 흔들기 테스트 영상(wave_example, 어깨 높이에서 흔듦)은 모든 모델이 박수로 판정. 손 흔들기 학습 영상을 5개 더 넣은 v4m도 못 맞힘. 테스트 영상이 1개(5초)뿐이라 판단 근거도 약함.
- 연속 영상에 그대로 돌리면 v3m이 박수를 너무 자주 냄: 손 들기·악수·팔벌려뛰기 영상에서도 clap 판정(아래 표). 모델이 팔이 몸 앞에서 움직임을 박수로 넓게 배운 것. 그래서 8011 실시간 화면에는 아직 GRU를 켜지 않았다.
- 해결 방향: 손 들기·악수·팔벌려뛰기처럼 네 동작이 아닌 움직임을 idle(또는 other) 라벨로 학습에 넣기, 웹캠으로 여러 사람 데이터를 직접 모으기(9-3 --record).

## 2. 규칙 개선 (pose_eval_v2.py로 측정, pose_app_v2.py에 같은 규칙 적용)

### 바꾼 것
- 동작 횟수를 동작 구간 단위로 셈: 기본 → 동작 → 기본 한 번이 1회 (v1은 한팔들기↔손흔들기가 오갈 때마다 따로 셈)
- 손흔들기: 손목 좌우 방향 전환 2번 → 3번, 그리고 1.5초 창 안 80% 이상 프레임에서 손이 올라가 있어야 함(팔을 내리는 동작 제외), 한 번 손흔들기면 1초 유지
- 악수: 두 사람 중심이 어깨 너비 0.8배 이상 떨어져 있고, 각자 손목을 상대 쪽으로 어깨 너비 0.3배 이상 뻗었을 때만 후보. 0.5초 이상 이어져야 1회, 같은 자리(어깨 너비 2배 안)는 2초 안에 다시 안 셈
- --low-wave 옵션: 어깨 높이에서 흔드는 손도 손흔들기로 봄

### 영상별 횟수 (정답은 장면 모음 이미지를 보고 사람이 대략 센 값)
| 영상 | 대략의 정답 | 규칙 v1 | 규칙 v2 | v2 + 어깨 높이 흔들기 |
| --- | --- | --- | --- | --- |
| wave_example.webm | 손흔들기 1 (어깨 높이) | 0 (0) | 0 (0) | 손흔들기 3 |
| children_waving.webm | - | 한팔들기 7, 손흔들기 2, 악수 5 (14) | 한팔들기 8 (8) | 한팔들기 7, 손흔들기 1 |
| raise_hand.webm | - | 0 (0) | 0 (0) | - |
| jumping_jacks_burpees.webm | 팔 올림 점프 약 6~8 (정면 0~21초 + 측면 22~45초) | 한팔들기 1, 만세 7, 손흔들기 1 (9) | 만세 7 (7) | - |
| handshake_tv.ogv | 악수 약 4~6 (0~20초 영접 장면) | 한팔들기 1, 악수 32 (33) | 한팔들기 1, 악수 2 (3) | - |
| applause_wikiconvention.webm | 악수 0 (청중 박수) | 한팔들기 1, 만세 2, 손흔들기 1, 악수 4 (8) | 한팔들기 2, 만세 2 (4) | - |
| clapping_nhs.webm | 동작 0 (박수) | 0 (0) | 0 (0) | 0 |
| mk28302_arms_crossed_man_green.mp4 | 동작 0 (팔짱만) | 0 (0) | 0 (0) | 0 |
| mk11653_arms_crossed_woman_standing.mp4 | 동작 0 (팔짱만) | 0 (0) | 0 (0) | - |
| mk29963_arms_crossed_woman_livingroom.mp4 | 동작 0 (팔짱만) | 0 (0) | 0 (0) | 0 |
| mk50643_arms_crossed_model.mp4 | - | 한팔들기 3, 손흔들기 3 (6) | 한팔들기 1, 손흔들기 2 (3) | - |
| mk35954_classroom_raise_hands_a.mp4 | - | 한팔들기 1, 만세 1 (2) | 만세 1 (1) | - |
| mk35968_classroom_raise_hands_b.mp4 | - | 한팔들기 4, 만세 1, 손흔들기 3 (8) | 손흔들기 1 (1) | - |
| mk11553_raise_hand_question.mp4 | - | 만세 1 (1) | 만세 1 (1) | - |
| mk33479_waving_two_boys.mp4 | 손흔들기 2 (두 명) | 한팔들기 20, 손흔들기 17 (37) | 한팔들기 1, 손흔들기 2 (3) | 한팔들기 1, 손흔들기 2 |
| mk48998_wave_elderly_couple.mp4 | - | 악수 1 (1) | 0 (0) | 손흔들기 1 |
| mk6389_wave_through_window.mp4 | - | 한팔들기 1, 손흔들기 2 (3) | 손흔들기 2 (2) | 손흔들기 2 |
| vtest.avi | 동작 0 (걷는 사람들) | 0 (0) | 0 (0) | 0 |

| 영상 | 학습에 씀 | action.pt @0.7 | action_v3m.pt @0.7 | action.pt @0.5 | action_v3m.pt @0.5 |
| --- | --- | --- | --- | --- | --- |
| wave_example.webm |  | 0 | clap 3 | arms_crossed 3 | clap 3 |
| children_waving.webm |  | 0 | clap 42, wave 1 | arms_crossed 3 | clap 43, wave 1 |
| raise_hand.webm |  | 0 | 0 | 0 | 0 |
| jumping_jacks_burpees.webm |  | 0 | clap 7, wave 4 | wave 1 | clap 10, wave 9 |
| handshake_tv.ogv |  | 0 | clap 11, wave 3 | 0 | clap 14, wave 9 |
| applause_wikiconvention.webm |  | 0 | clap 8 | arms_crossed 3 | clap 9, wave 1 |
| clapping_nhs.webm | 예 | 0 | clap 1 | arms_crossed 1, wave 1 | clap 2, wave 1 |
| mk28302_arms_crossed_man_green.mp4 | 예 | 0 | arms_crossed 1 | 0 | arms_crossed 1 |
| mk11653_arms_crossed_woman_standing.mp4 | 예 | 0 | arms_crossed 1 | 0 | arms_crossed 1 |
| mk29963_arms_crossed_woman_livingroom.mp4 |  | 0 | arms_crossed 1 | 0 | arms_crossed 1 |
| mk50643_arms_crossed_model.mp4 |  | 0 | arms_crossed 2, clap 1, wave 3 | 0 | arms_crossed 2, clap 1, wave 4 |
| mk35954_classroom_raise_hands_a.mp4 |  | 0 | clap 5, wave 2 | 0 | arms_crossed 2, clap 6, wave 4 |
| mk35968_classroom_raise_hands_b.mp4 |  | 0 | clap 6, wave 1 | 0 | arms_crossed 2, clap 8, wave 4 |
| mk11553_raise_hand_question.mp4 |  | 0 | clap 2 | 0 | arms_crossed 1, clap 3, wave 1 |
| mk33479_waving_two_boys.mp4 | 예 | 0 | wave 2 | 0 | wave 2 |
| mk48998_wave_elderly_couple.mp4 |  | 0 | clap 2 | 0 | clap 2 |
| mk6389_wave_through_window.mp4 |  | 0 | clap 1 | 0 | clap 1 |
| vtest.avi |  | 0 | clap 1 | 0 | clap 1 |

### 해석
- 악수 TV: v1 악수 32회 → v2 2회 (정답 약 4~6). 과다 카운트는 해결됐지만 이번엔 덜 셈. 영접 장면에서 여러 사람이 겹쳐 서서 두 사람이 떨어져 있어야 함 조건에 걸리는 경우가 있음. 기준(0.8배)을 0.5배로 낮추는 것을 다음에 시험.
- 청중 박수: v1 악수 4회 → v2 0회. 박수를 악수로 오인하던 문제 해결.
- 두 아이 손 흔들기: v1 37회(한팔들기↔손흔들기 깜빡임) → v2 3회 (정답 2). 가장 크게 좋아짐.
- 팔벌려뛰기: v1 9회 → v2 만세 7회 (정답 약 6~8). 팔 내리는 동작의 손흔들기 오판정 1회와 한팔들기 1회가 없어짐.
- 교실 손 들기 b: v1 8회 → v2 1회. 여러 아이가 차례로 손을 드는 영상이라 v2가 덜 세는 쪽(추적 ID가 이어지면 한 구간으로 봄).
- 팔짱·박수·걷는 사람 영상: v1·v2 모두 0회로 오판정 없음.
- --low-wave: 어깨 높이 손 흔들기(wave_example)가 0 → 3회(정답 1, 끊겼다 이어지며 더 셈), 노부부 인사 0 → 1회. 팔짱·박수·걷는 사람 영상 5개에서 오판정 0회라 켜도 안전한 편.

## 3. 만든 파일
- 코드: extract_feats.py, eval_gru.py, train_pose_v2.py(--mirror), pose_eval_v2.py, pose_app_v2.py(--low-wave), runs/step9_improve/summarize_improve.py
- 데이터: data/action_jiwon/(정지원님 CSV 복사본), data/action_v2·v3·v4/, data/action_test/
- 모델: models/action_v2.pt, action_v3.pt, action_v3m.pt(추천), action_v4m.pt
- 기록: train_v2_pose.log, train_v3_pose.log, train_v3m_pose.log, train_v4m_pose.log, runs/step9_improve/v2.jsonl, v2_lowwave.jsonl

## 4. 측면 시점 (WVU Multi-view Action Dataset, 2026-10-26)

### 출처 찾기
| 출처 | 결과 |
| --- | --- |
| WVU Multi-view Action Dataset (웨스트버지니아대, 연구용 공개) | 성공. 카메라 8대가 빙 둘러 찍은 장면 사진. 동작 zip(약 615MB)을 통째로 받지 않고 HTTP Range로 필요한 장면만 꺼냄(remote_zip.py, 214MB). 20fps 영상으로 묶음(runs/step9_improve/wvu_build.py) |
| UTD-MHAD MultiView (텍사스대) | HTTP/2에서 몇 MB마다 끊기고 처음부터 다시 받아져 실패 → HTTP/1.1 Range 8MB 조각 6개 병렬로 받아 성공(186MB, 압축 검사 통과). 그런데 안에 RGB 영상이 없고 깊이·관절·관성 센서(.mat)만 있어 사용 못 함 |
| Mixkit, Wikimedia Commons | 측면 손 들기 영상 없음(대부분 정면) |

사용한 WVU 영상: 동작 1 가만히 서 있기, 3 박수, 4 한 손 흔들기, 5 두 손 흔들기, 8 팔벌려뛰기 × 사람 3명(part1 train1~3) × 시점 4개(V1 정면, V2 45도, V3·V7 측면 90도), 각 71프레임(약 3.5초) = 60개.

### 시점에 따라 어깨 너비가 얼마나 줄어드나 (runs/step9_improve/wvu_view.jsonl)
| 시점 | 어깨 너비 / 몸통 길이 (중앙값, 동작 5개 범위) |
| --- | --- |
| V1 정면 | 0.60 ~ 0.62 |
| V2 45도 | 0.49 ~ 0.59 |
| V3 측면 | 0.07 ~ 0.30 |
| V7 반대 측면 | 0.14 ~ 0.18 |

측면에서는 어깨 너비가 정면의 1/2 ~ 1/8로 줄어든다. 규칙의 기준(손목이 어깨보다 어깨 너비 0.3배 위, 악수 어깨 너비 0.5배 이내, 손 흔들기 어깨 너비 0.15배 이동)이 모두 어깨 너비 단위라, 측면에서는 기준이 거의 0이 되어 아주 작은 움직임도 넘거나(오판정) 좌우 흔들기가 화면에서 앞뒤 움직임이 되어 안 잡힌다. README 9-2에서 말한 한계가 숫자로 확인됨. (어깨 검출 자체는 모든 시점에서 100%)

### 규칙 v1 / v2 / GRU 횟수 (사람 3명 합계)
| 동작 (3명 합계) | 기대 | 시점 | 규칙 v1 | 규칙 v2 | v2 + 어깨 높이 흔들기 | GRU action.pt @0.5 | GRU action_v3m.pt @0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 가만히 서 있기 | 0 | V1 정면 | 0 | 0 | 0 | arms_crossed 3 | 0 |
| 가만히 서 있기 | 0 | V2 45도 | 0 | 0 | 0 | arms_crossed 2 | 0 |
| 가만히 서 있기 | 0 | V3 측면 | 0 | 0 | 0 | arms_crossed 1 | arms_crossed 3 |
| 가만히 서 있기 | 0 | V7 반대 측면 | 0 | 0 | 0 | 0 | 0 |
| 박수 | 규칙 0 / GRU clap | V1 정면 | 0 | 0 | 0 | arms_crossed 3 | arms_crossed 2 |
| 박수 | 규칙 0 / GRU clap | V2 45도 | 0 | 0 | 0 | arms_crossed 4, clap 1 | clap 3 |
| 박수 | 규칙 0 / GRU clap | V3 측면 | 0 | 0 | 0 | 0 | arms_crossed 3 |
| 박수 | 규칙 0 / GRU clap | V7 반대 측면 | 0 | 0 | 0 | 0 | arms_crossed 1 |
| 한 손 흔들기 | 손흔들기 3 | V1 정면 | 한팔들기 3, 손흔들기 3 | 한팔들기 3 | 손흔들기 3 | arms_crossed 3 | wave 3 |
| 한 손 흔들기 | 손흔들기 3 | V2 45도 | 한팔들기 3, 손흔들기 3 | 한팔들기 3, 손흔들기 1 | 손흔들기 3, 한팔들기 1 | arms_crossed 3 | clap 2, wave 1 |
| 한 손 흔들기 | 손흔들기 3 | V3 측면 | 손흔들기 2, 한팔들기 1 | 한팔들기 2 | 한팔들기 1, 손흔들기 1 | arms_crossed 1 | arms_crossed 3, clap 1 |
| 한 손 흔들기 | 손흔들기 3 | V7 반대 측면 | 한팔들기 1, 손흔들기 1 | 한팔들기 1 | 한팔들기 1 | 0 | clap 1 |
| 두 손 흔들기 | 손흔들기 또는 만세 3 | V1 정면 | 만세 3 | 만세 3 | 만세 2, 손흔들기 1 | wave 1, arms_crossed 1 | wave 3 |
| 두 손 흔들기 | 손흔들기 또는 만세 3 | V2 45도 | 만세 3 | 만세 3 | 만세 2, 손흔들기 1 | arms_crossed 3 | wave 3 |
| 두 손 흔들기 | 손흔들기 또는 만세 3 | V3 측면 | 만세 2, 손흔들기 1 | 만세 2, 손흔들기 1 | 만세 2, 손흔들기 1 | arms_crossed 1, clap 1 | clap 4, arms_crossed 1, wave 1 |
| 두 손 흔들기 | 손흔들기 또는 만세 3 | V7 반대 측면 | 만세 3, 손흔들기 3 | 만세 1, 손흔들기 1, 한팔들기 1 | 만세 1, 손흔들기 1, 한팔들기 1 | 0 | wave 2, clap 1 |
| 팔벌려뛰기 | 만세 여러 번 | V1 정면 | 0 | 0 | 0 | arms_crossed 2 | clap 3 |
| 팔벌려뛰기 | 만세 여러 번 | V2 45도 | 0 | 0 | 0 | arms_crossed 3 | clap 3 |
| 팔벌려뛰기 | 만세 여러 번 | V3 측면 | 0 | 0 | 0 | clap 2, arms_crossed 1 | clap 3 |
| 팔벌려뛰기 | 만세 여러 번 | V7 반대 측면 | 0 | 0 | 0 | 0 | clap 3, wave 1 |

### 해석 (규칙)
- 한 손 흔들기: 정면에서 v1은 한팔들기 3 + 손흔들기 3(같은 동작을 두 번 셈), v2는 한팔들기 3(흔들기 기준 3번 전환을 3.5초 안에 못 채움), v2 + 어깨 높이 흔들기는 손흔들기 3으로 정답. 측면에서는 손흔들기가 크게 줄어듦: V3에서 v1 2/3, v2 0/3, v2 + 어깨 높이 1/3, V7에서 v1 1/3, v2·v2 + 어깨 높이 0/3. 좌우로 흔드는 손이 측면 카메라에는 앞뒤 움직임이 되어 x 좌표가 조금만 바뀌고, 방향 전환 기준을 3번으로 올린 v2가 더 많이 놓친다.
- 두 손 흔들기: 정면·45도는 모두 만세 3으로 일정. V7 측면에서 v1은 만세 3 + 손흔들기 3으로 두 번 셌고 v2는 3회로 줄었다.
- 팔벌려뛰기: 모든 시점·모든 규칙에서 0. 손목이 올라간 프레임은 17~21%로 있지만, 팔이 머리 위에 있는 시간이 0.6초 다수결 창보다 짧아 smooth 판정에서 사라진다. 빠른 동작에는 0.6초 안정화가 오히려 방해(raw로는 잡힘).
- 서 있기·박수: 모든 시점에서 규칙 오판정 0.

### 측면 데이터로 GRU 다시 학습 (action_v5m.pt)
v3m 학습 데이터 + WVU 사람 1·2(모든 시점)의 서 있기·박수·손 흔들기, 좌우 뒤집기 증강. 테스트는 학습에 안 쓴 사람 3을 시점별로(eval_gru.py, 윈도 30).

| 시점 (사람 3) | action.pt | action_v3m.pt | action_v5m.pt |
| --- | --- | --- | --- |
| V1 정면 | 6% | 36% | 86% |
| V2 45도 | 0% | 75% | 100% |
| V3 측면 | 25% | 8% | 100% |
| V7 반대 측면 | 25% | 19% | 53% (서 있기를 손 흔들기로 착각) |
| 앞의 스톡 영상 테스트(팔짱·박수·기본·손 흔들기) | 34% | 69% | 54% |

- 측면 데이터를 넣으니 측면 정확도가 8% → 100%(V3)로 크게 올랐다. 다만 WVU는 같은 방·같은 배경이라 사람만 바뀐 테스트여서 실제보다 좋게 나왔을 가능성이 크다.
- 대신 스톡 영상 테스트에서는 69% → 54%로 내려갔다: 손 흔들기는 0% → 77%로 좋아졌지만 팔짱이 100% → 21%로 떨어짐(팔짱을 박수로 착각). WVU에 팔짱 데이터가 없어 박수·흔들기 데이터 비중이 커졌기 때문. 팔짱 측면 데이터를 더하면 해결될 것으로 보인다.
- 결론: 측면 데이터가 측면 인식에 결정적이라는 것은 확인. 용도에 따라 정면 위주면 v3m, 여러 각도 카메라면 v5m. 둘 다 아직 실시간 화면에는 켜지 않음.

### 이번에 만든 파일
remote_zip.py, runs/step9_improve/{wvu_build.py, wvu_view.py, summarize_wvu.py, run_wvu.sh, extract_wvu.sh, wvu.jsonl, wvu_lowwave.jsonl, wvu_view.jsonl, wvu_gru.log}, media/pose/wvu/(영상 60개, frames/), data/action_v5/, data/action_test_wvu/, models/action_v5m.pt, train_v5m_pose.log, media/pose/utd_mhad/MultiViewDataset.zip(RGB 없음)

## 5. 손 흔들기 개선 (2026-10-26)

### 평가 세트 (고정)
- 손 흔들기(양성) 스톡 7개: wave_example, mk33479, mk48998, mk6389(뒤·옆모습), mk46150, mk48679, children_waving. 정답은 장면 모음 이미지로 사람이 셈. mk1305(드론, 사람이 점), mk48680(손이 화면 밖)은 제외.
- 손 흔들기 WVU 24개: 한 손·두 손 흔들기 × 사람 3명 × 시점 4개.
- 음성 스톡 18개: 팔짱 4, 박수 6, 걷기·서 있기 3, 교실·질문 손 들기 3, 팔벌려뛰기, 악수 TV. 음성 WVU 36개: 서 있기·박수·팔벌려뛰기 × 3명 × 4시점.
- 지표: 양성 검출(1회 이상 잡음), 횟수 오차, 음성 오판정(잘못 센 손흔들기 수). 비교는 pose_eval_v3.py로 같은 검출 결과에서 여러 방식을 한 번에 셈.

### 비교한 방식
| 방식 | 내용 |
| --- | --- |
| v2 | 지금까지 쓰던 방식: 손목이 어깨보다 어깨 너비 0.3배 위 + 손목 x 방향 전환 3번 + 손 올림 80% |
| v2low | v2 + 어깨 높이 흔들기 허용(--low-wave) |
| traj | 팔마다, 흔드는 자세(손목이 팔꿈치보다 위, 어깨선보다 몸통 0.25배 아래까지)가 70% 이상 + 팔꿈치 기준 손목 궤적(아래팔 길이로 나눔)의 주 방향으로 0.2 이상씩 방향 전환 2번 |
| angle | 같은 자세 조건 + 아래팔 각도가 15도 이상씩 방향 전환 2번 |
| angle3w | angle을 방향 전환 3번, 창 2초로 엄격하게 |
| trajB / angleB | traj / angle + 두 손목이 붙은 프레임(박수 자세) 제외 + 진폭 0.3 / 20도 + 자세 80% |
| trajC / angleC | traj / angle + 박수 자세 제외만 (나머지는 traj / angle과 같음) |

모든 방식: 1.5초 창(angle3w는 2초), 창 안 방향 전환 15번 초과(약 5Hz 넘는 떨림)는 제외, 손흔들기면 1초 유지. 1·2차에서 함께 본 traj3·xnorm은 오판정이 많아 3차 표에서 뺐다(xnorm 음성 오판정 57).

#### 손 흔들기 영상 (양성)
| 영상 | 정답 | v2 | v2low | traj | angle | angle3w | trajB | angleB | trajC | angleC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wave_example.webm | 1 (3명 중 왼쪽 1명이 어깨 높이에서 흔듦) | 0 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| mk33479_waving_two_boys.mp4 | 1 (여자아이 1명이 5~13초 흔듦) | 3 | 3 | 2 | 2 | 2 | 3 | 1 | 2 | 2 |
| mk48998_wave_elderly_couple.mp4 | 2 (2명이 내내 흔듦) | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| mk6389_wave_through_window.mp4 | 1 (뒤·옆모습으로 창밖에 흔듦) | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| mk46150_wave_man_couch.mp4 | 1 (2~5초 흔듦) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk48679_wave_woman_videocall.mp4 | 1 (얼굴 가까이, 2~3초 흔듦) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| children_waving.webm | 1 (여러 아이, 1회 이상이면 정답) | 0 | 1 | 4 | 2 | 1 | 2 | 2 | 3 | 2 |
| **검출 (7개 중 1회 이상)** | | 2 | 5 | 5 | 4 | 4 | 4 | 4 | 5 | 4 |
| **횟수 오차 합** | | 9 | 8 | 10 | 8 | 7 | 9 | 7 | 9 | 8 |

#### 손 흔들기가 아닌 영상 (음성, 숫자 = 잘못 센 손흔들기)
| 영상 | v2 | v2low | traj | angle | angle3w | trajB | angleB | trajC | angleC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mk28302_arms_crossed_man_green.mp4 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| mk11653_arms_crossed_woman_standing.mp4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk29963_arms_crossed_woman_livingroom.mp4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk50643_arms_crossed_model.mp4 | 2 | 4 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| clapping_nhs.webm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| applause_wikiconvention.webm | 2 | 3 | 3 | 3 | 3 | 1 | 1 | 2 | 2 |
| mk23085_clap_office.mp4 | 0 | 0 | 2 | 2 | 1 | 0 | 0 | 1 | 0 |
| mk48667_clap_group_therapy.mp4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| mk5953_clap_seminar.mp4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk24214_clap_conference.mp4 | 0 | 0 | 2 | 2 | 2 | 3 | 3 | 2 | 2 |
| mk4519_idle_students_walking.mp4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk460_idle_standing_woman.mp4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vtest.avi | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mk35954_classroom_raise_hands_a.mp4 | 0 | 0 | 2 | 2 | 1 | 1 | 0 | 1 | 1 |
| mk35968_classroom_raise_hands_b.mp4 | 1 | 1 | 3 | 3 | 1 | 2 | 0 | 2 | 2 |
| mk11553_raise_hand_question.mp4 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| jumping_jacks_burpees.webm | 1 | 2 | 3 | 3 | 3 | 0 | 0 | 1 | 1 |
| handshake_tv.ogv | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 2 | 2 |
| **오판정 합 (오판정 난 영상 수/18)** | 6 (4) | 10 (4) | 21 (11) | 19 (9) | 12 (7) | 7 (4) | 4 (2) | 13 (9) | 10 (6) |

#### WVU 시점별 (손 흔들기 = 한 손·두 손 흔들기 6개 영상 중 잡은 수, 음성 = 서 있기·박수·팔벌려뛰기 9개 영상의 오판정 수)
| 시점 | 구분 | v2 | v2low | traj | angle | angle3w | trajB | angleB | trajC | angleC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 정면 | 손 흔들기 검출 /6 | 2 | 5 | 6 | 6 | 6 | 5 | 5 | 6 | 6 |
| V1 정면 | 음성 오판정 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V2 45도 | 손 흔들기 검출 /6 | 4 | 6 | 6 | 6 | 5 | 6 | 6 | 6 | 6 |
| V2 45도 | 음성 오판정 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V3 측면 | 손 흔들기 검출 /6 | 2 | 3 | 6 | 6 | 1 | 1 | 1 | 5 | 4 |
| V3 측면 | 음성 오판정 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V7 반대 측면 | 손 흔들기 검출 /6 | 1 | 1 | 3 | 3 | 3 | 2 | 2 | 3 | 3 |
| V7 반대 측면 | 음성 오판정 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 전체 | 손 흔들기 검출 /24 | 9 | 15 | 21 | 21 | 15 | 14 | 14 | 20 | 19 |
| 전체 | 음성 오판정 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

#### v2가 놓친 이유 (양성 영상 진단)
| 영상 | 프레임 | 어깨 안 보임 | 손이 v2 기준 높이 위 | 손목이 팔꿈치 위(흔드는 자세) | v2 창 안 최대 방향 전환 | 어깨/몸통 |
| --- | --- | --- | --- | --- | --- | --- |
| wave_example.webm | 468 | 0% | 0% | 99% | 0 | 0.58 |
| mk33479_waving_two_boys.mp4 | 634 | 0% | 55% | 80% | 8 | 0.54 |
| mk48998_wave_elderly_couple.mp4 | 886 | 0% | 1% | 88% | 0 | 0.4 |
| mk6389_wave_through_window.mp4 | 570 | 0% | 32% | 60% | 24 | 0.4 |
| mk46150_wave_man_couch.mp4 | 300 | 0% | 0% | 37% | 0 | 0.8 |
| mk48679_wave_woman_videocall.mp4 | 302 | 0% | 0% | 0% | 0 | 0.4 |
| children_waving.webm | 4148 | 22% | 3% | 12% | 3 | 0.4 |

### v2가 놓친 이유
- 손 높이: wave_example·노부부·소파 남성·영상통화 여성은 손목이 v2 기준 높이(어깨보다 어깨 너비 0.3배 위)를 넘은 프레임이 0~1%. 사람들은 대부분 어깨 높이에서 흔든다. 반면 손목이 팔꿈치보다 위인 프레임은 37~99%.
- 측면: 좌우로 흔드는 손이 측면 카메라에는 앞뒤 움직임이라 손목 x가 거의 안 바뀜. 아래팔 각도·궤적으로 보면 V3 측면 2/6 → 4~6/6.
- 영상통화 여성(mk48679)은 손목이 팔꿈치보다 위인 프레임도 0%: 얼굴 가까이 손만 나와 팔꿈치가 안 잡힘 → 모든 방식이 놓침. 소파 남성(mk46150)도 흔드는 시간이 짧고 흔드는 자세 비율이 37%라 모든 방식이 놓침.

### 선택한 설정: angleC (pose_app_v3.py 기본값)
| | v2 (이전) | angleC (새 기본) | trajC | angleB |
| --- | --- | --- | --- | --- |
| 스톡 손 흔들기 검출 /7 | 2 | 4 | 5 | 4 |
| WVU 손 흔들기 검출 /24 | 9 | 19 | 20 | 14 |
| 측면 V3 검출 /6 | 2 | 4 | 5 | 1 |
| 스톡 음성 오판정 (오판정 난 영상 수) | 6 (4) | 10 (6) | 13 (9) | 4 (2) |
| WVU 음성 오판정 | 0 | 0 | 0 | 0 |

- angleC: 양성 검출이 31개 중 11개 → 23개로 두 배, 대신 스톡 음성 오판정이 6 → 10회로 조금 늘어 균형이 가장 좋다. 오판정은 군중 박수·회의 박수(박수 중 한 손을 드는 장면), 교실 손 들기, 팔벌려뛰기에서 남.
- 오판정을 최소로 하려면 angleB(검출 18/31, 오판정 4), 검출을 최대로 하려면 trajC(25/31, 오판정 13). pose_app_v3.py 위쪽 WAVE3_KIND·WAVE3_AMP로 바꿀 수 있고, --wave v2로 예전 방식도 쓸 수 있다.

### GRU 손 흔들기 개선 시도
- action_v6m.pt: v3m 데이터 + 손 흔들기(Mixkit 2개, WVU 사람 1·2의 한 손·두 손 흔들기) + 박수(WVU 사람 1·2) + 그 밖의 움직임을 idle로(WVU 서 있기·팔벌려뛰기, 악수 TV, 교실 손 들기), 좌우 뒤집기.
- action_v6u.pt: v6m과 같은 데이터, 하체 관절(엉덩이·무릎·발목)의 좌표·신뢰도를 지우고 상반신만 사용(train_pose_v3.py --upper). 박수 학습 영상은 앉은 청중이라 하체가 안 보이고 손 흔들기 테스트 영상도 상반신만 나와, 모델이 "하체 안 보임 = 박수"로 배운 것 같아서 시험.
- 테스트: 스톡(팔짱 mk29963, 박수 applause, 기본 vtest, 손 흔들기 wave_example·mk46150·mk48679 3개 영상) / WVU 사람 3 시점별. 모두 학습에 안 씀.

| 모델 | 스톡 팔짱 | 스톡 박수 | 스톡 기본 | 스톡 손 흔들기 | 스톡 전체 | WVU V1 | V2 | V3 | V7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_v3m.pt | 100% | 100% | 76% | 0% | 49% | 36% | 75% | 8% | 19% |
| action_v5m.pt | 21% | 62% | 71% | 29% | 38% | 86% | 100% | 100% | 53% |
| action_v6m.pt | 55% | 25% | 100% | 11% | 41% | 83% | 100% | 100% | 61% |
| action_v6u.pt (상반신) | 7% | 50% | 100% | 48% | 48% | 72% | 94% | 97% | 78% |

이어지는 영상에 그대로 돌렸을 때(pose_eval_v2.py, 확률 0.7 이상 판정 횟수):

| 영상 | 정답 | action_v3m.pt @0.7 | action_v5m.pt @0.7 | action_v6m.pt @0.7 |
| --- | --- | --- | --- | --- |
| mk29963_arms_crossed_woman_livingroom.mp4 | arms_crossed | arms_crossed 1 | clap 3, arms_crossed 2 | arms_crossed 4, clap 4 |
| applause_wikiconvention.webm | clap | clap 8 | clap 7, wave 3 | clap 6, wave 4 |
| vtest.avi | idle(4동작 아님) | clap 1 | wave 2 | 0 |
| mk35968_classroom_raise_hands_b.mp4 | idle(4동작 아님) | clap 6, wave 1 | wave 11, clap 6 | wave 1, clap 1 |
| mk11553_raise_hand_question.mp4 | idle(4동작 아님) | clap 2 | clap 4, wave 2 | clap 4, wave 2 |
| jumping_jacks_burpees.webm | idle(4동작 아님) | clap 7, wave 4 | wave 9, clap 8 | wave 7, clap 9 |
| e8_s3_v1.avi | idle(4동작 아님) | clap 1 | wave 1 | 0 |
| e8_s3_v3.avi | idle(4동작 아님) | 0 | wave 1, clap 1 | 0 |
| mk46150_wave_man_couch.mp4 | wave 1 | clap 1 | clap 1 | clap 1 |
| mk48679_wave_woman_videocall.mp4 | wave 1 | clap 1 | clap 1 | clap 2, wave 1 |
| wave_example.webm | wave 1 | clap 3 | wave 10, clap 7 | clap 3 |

- 상반신만 쓰니 스톡 손 흔들기가 11% → 48%로 올라 "하체 구도" 추정이 일부 맞았다. 하지만 팔짱이 7%로 떨어짐(팔짱·박수 모두 손목이 가슴 앞에 모인 자세라 상반신만으로는 헷갈림).
- v6m은 idle(4동작이 아닌 움직임)을 스톡에서 100% 맞혀 "그 밖의 움직임" 학습은 효과가 있었지만, 이어지는 영상에서는 교실 손 들기·팔벌려뛰기·군중 박수에서 여전히 clap·wave를 내고, 손 흔들기 영상 3개는 대부분 clap으로 판정.
- 결론: GRU는 학습 영상이 동작마다 2~6개뿐이라 촬영 구도에 따라 크게 흔들린다. 지금은 규칙(angleC)이 더 믿을 만해서 8011 화면에 GRU(--model)는 켜지 않았다. 손목 속도를 특징에 더한 버전은 특징 형식(53차원)과 pose_app 전체를 바꿔야 해서 이번에는 하지 않음.

### 8011 화면
- pose_app_v3.py(손흔들기 angleC) 실행 중. 영상: 소파 남성 손 흔들기 → WVU 한 손 흔들기 정면·측면 → 노부부 손 흔들기 → 사무실 박수 반복. 1분간 손흔들기 2회, 한팔들기 4회, 악수 3회 기록, 약 15 fps.
- 실시간 화면에서 새로 본 문제: 노부부처럼 두 사람이 나란히 앉아 함께 손을 흔들면 악수로 잘못 셈(두 사람의 손목이 가까움). 앉은 자세에서는 흔드는 손도 악수 높이(가슴~허리) 조건에 들어감.
- 이전 실행 명령은 runs/step9_improve/original_sessions.txt에 기록.

### 남은 한계
- 영상통화처럼 팔꿈치가 화면 밖이면 어떤 방식도 못 잡음(손만 보이는 장면은 손 관절 모델이 필요).
- 반대 측면(V7)은 모든 방식이 3/6: 흔드는 팔이 몸에 가려짐.
- 뒷모습·옆모습으로 창밖에 흔드는 영상(mk6389)은 angleC가 놓침(v2·trajC는 잡음): 뒤에서 보면 아래팔 각도 변화가 작게 보임. 이런 카메라 위치가 많으면 trajC가 낫다.
- 박수 중 손을 드는 장면, 여러 아이가 손을 드는 교실 장면은 손 흔들기와 구분이 어려움.
- 첫 실행 때 pose_app_v3.py에 변수 이름 충돌(정면 점수 fa) 버그가 있어 바로 고침.

### 이번에 만든 파일
pose_eval_v3.py, pose_app_v3.py, train_pose_v3.py(--upper), eval_gru.py(상반신 모델 지원 추가, 이전 버전은 runs/step9_wave/eval_gru_before_upper.py), models/action_v6m.pt·action_v6u.pt, data/action_v6/, data/action_test_v6/, runs/step9_wave/(1~3차 비교 jsonl, summary*.md, gru_v6.log, gru_stream.md, 스크립트), train_v6m_pose.log, train_v6u_pose.log, pose_v3_video.log

## 6. 악수 오판정 해결과 반복 개선 (2026-10-27)

### 6-1. 악수 오판정

#### 원인 다시 찾기
- 5절에서 "노부부처럼 나란히 앉아 함께 손을 흔들면 악수로 셈"이라고 적었는데 틀렸다. 실제 앱(pose_app_v3.py)에 영상을 하나씩 반복 재생해 재현해 보니(runs/step9_hs/live_repro.sh) 노부부 영상은 2번 반복에 악수 0회, 손흔들기 6회였다.
- 실제 원인은 같은 재생 목록의 사무실 박수 영상(mk23085): 나란히 서서 박수 치는 옆 사람끼리(#1+#3, #4+#6) 2번 반복에 악수 5회.
- 영상을 따로 돌리는 비교(pose_eval_v4.py)에서는 같은 영상이 악수 1회로, 실제 앱(초당 약 13~15장, 스트림 재압축)과 결과가 달랐다. 그래서 최종 판단은 실제 앱 재현으로 했다.

#### 악수를 잘 못 잡는 이유 (측면)
- 악수는 보통 옆으로 마주 서서 하므로 카메라에는 측면으로 찍힌다. 측면에서는 어깨 너비가 좁게 보여, 어깨 너비 단위 기준(손목 거리 0.5배 등)이 너무 엄격해진다. 영접 악수(mk25253)는 두 손목이 가장 가까울 때도 어깨 너비의 2.2배였다.
- 측면용 단위(max(어깨 너비, 몸통 0.4배))로 바꾼 hsE·hsF도 영접·협상 후 악수는 0회: 맞잡은 손에서 뒤쪽 사람 손목이 가려져 관절 위치가 크게 어긋난다. 기준을 0.8배로 풀면(hsF) 걷는 학생 영상에서 오판정 7회.
- Mixkit 악수 영상 8개 중 4개(mk17231·30006·30015·45932)는 손만 크게 나오는 클로즈업이라 사람 관절이 안 잡혀 평가에서 뺐다.

#### 영상을 따로 돌린 비교 (pose_eval_v4.py, 프레임을 건너뛰지 않음, 악수 횟수)
| 영상 | 기대 | hs2(이전) | hsD | hsG | hsH |
| --- | --- | --- | --- | --- | --- |
| handshake_tv.ogv | 악수 약 4~6 | 2 | 2 | 2 | 1 |
| mk49419_hs_yacht.mp4 | 악수 1 | 1 | 1 | 1 | 1 |
| mk23387_hs_after_negotiation.mp4 | 악수 약 3 | 0 | 0 | 0 | 0 |
| mk25253_hs_welcoming_partner.mp4 | 악수 1 | 0 | 0 | 0 | 0 |
| mk23085_clap_office.mp4 | 0 | 1 | 1 | 1 | 1 |
| mk36153_clap_standing_meeting.mp4 | 0 | 3 | 2 | 2 | 0 |
| mk4519_idle_students_walking.mp4 | 0 | 1 | 1 | 1 | 1 |
| 노부부·wave_example·아이들 손 흔들기·군중 박수 4개·교실 손 들기·vtest | 0 | 0 | 0 | 0 | 0 |
- hsD: 간격 기준 완화(어깨 너비 0.5배) + 맞잡은 손목이 팔꿈치보다 많이 위가 아님(손 흔들기·손 들기 제외) + 맞잡은 손이 두 사람 사이 + 손흔들기 중인 사람 제외
- hsG: hsD + 그 순간 박수 자세(맞잡은 손이 자기 다른 손과 붙음)면 제외 → 효과 없음(박수는 손을 붙였다 뗌)
- hsH: hsD + '박수 치는 사람'(최근 1.5초의 25% 이상 두 손을 붙임) 제외
- 실시간처럼 2프레임마다 1장만 처리해도(--stride 2) 경향은 같음(runs/step9_hs/hs4_s2.jsonl).

#### 실제 앱 재현 (영상 2번 반복, 악수 횟수)
| 앱 | 사무실 박수 (0이 정답) | 서서 하는 회의 박수 (0) | 악수 TV (양성) | 요트 (양성) | 영접 악수 (양성) |
| --- | --- | --- | --- | --- | --- |
| pose_app_v3.py (이전) | 5 | - | - | - | - |
| v4 1차 (hsH 조건) | 5 | 2 | 3 | 2 | - |
| v4 2차 (+ 두 사람이 코 방향으로 서로를 향함) | 0 | 0 | 0 | 0 | 0 |
| **v4 최종 (+ 두 사람이 모두 카메라 정면을 보면 제외)** | **0** | 2 | **3** | 0 | 0 |
- 2차(코 방향)는 오판정을 모두 없앴지만 진짜 악수도 모두 놓쳤다(사진 찍듯 카메라를 보며 악수하는 장면이 많음).
- 최종: 나란히 서서 카메라를 보며 박수 치는 사람들은 둘 다 정면 점수가 높아 걸러진다. 사무실 박수 오판정 5 → 0, 악수 TV 3회 유지. pose_app_v4.py 기본값.
- 남은 한계: 서서 하는 회의 박수(2회 오판정), 요트·영접 악수(놓침). 측면 악수는 손목이 가려지는 문제라 관절 규칙만으로는 한계가 있다.

### 6-2. 정정: wave_example
- 5절에서 wave_example의 정답을 "3명 중 1명"이라 적었는데, 장면을 다시 보니 3명 모두 어깨 높이에서 손을 흔든다. 정답은 3회이고, 규칙 angleC의 3회가 맞았다. (5절 양성 표의 횟수 오차도 wave_example 기준으로는 angleC가 0)

### 6-3. GRU 반복 개선 (runs/step9_loop/round_1~5.md)
종합 점수 = (스톡 테스트 동작별 정확도 평균 + WVU 사람 3 시점별 정확도 평균) / 2. 테스트 영상·사람은 학습에 넣지 않음.

| 라운드 | 가설·변경 | 결과 | 최고 |
| --- | --- | --- | --- |
| 1 | 학습 때 하체를 지운 복사본을 함께 학습(train_pose_v4.py --lowdrop), 클래스 균형(--balance) | 손 흔들기 11 → 45%지만 팔짱 7%. 종합 64.9 / 62.3 | v6u 68.3 유지 |
| 2 | Mixkit 영상 20개 추가(팔짱 3, 박수 8, 손 흔들기 6, 기본 3) → data/action_r2 | **action_r2m: 팔짱 100%, 기본 100%, WVU 평균 90.5%(반대 측면 61 → 81%)** | **77.1** |
| 3 | 손 흔들기 테스트 라벨 정리(흔드는 구간만, media/pose/test_clips) 후 전체 재평가 | 팔짱과 손 흔들기가 맞바뀜: 전신 모델 r2m은 손 흔들기 0%, 하체를 지운 모델은 손 흔들기 88%·팔짱 7% | r2m 75.8 (새 테스트 기준) |
| 4 | 결합: 손 흔들기=규칙 angleC, 팔짱·박수=GRU r2m (영상 전체로 비교) | 기대와 다른 판정 57(r2m 단독) → 45(결합). 팔벌려뛰기·악수·교실 손 들기의 박수 오판정은 남음. 30fps 영상을 15fps 학습 모델에 그대로 넣은 측정이라 라운드 5에서 실제 앱으로 다시 잼 | 개선 없음 |
| 5 | 실제 앱(pose_app_v4.py --model action_r2m.pt)으로 재현 | 팔짱 1(정답)·걷기 0은 맞지만 팔벌려뛰기 clap 13·wave 7, 악수 TV clap 11·wave 4, wave_example clap 8 → 규칙만 쓸 때보다 훨씬 나쁨 | 개선 없음 |
- 라운드 3·4·5 연속 개선 없음 → 기준대로 반복을 멈춤(약 3시간).
- 결론: 가장 좋은 GRU는 action_r2m.pt(팔짱·기본 강함, 측면 강함)지만, 실제 앱에서는 4동작이 아닌 움직임(팔벌려뛰기, 악수, 손 들기)을 박수·손 흔들기로 자주 착각해 8011 화면에는 켜지 않았다. 동작마다 학습 영상이 여전히 수 개~십수 개뿐이고, 촬영 구도(하체가 보이는지)에 따라 크게 흔들린다.
- 다음에 해 볼 것: 웹캠으로 반 학생 여러 명의 동작을 직접 모으기(9-3 --record), "그 밖의 움직임" 데이터를 크게 늘리기, 손목 속도 같은 특징 추가(특징 형식·앱을 함께 바꿔야 함).

### 6-4. 8011 화면
- pose_app_v4.py(손흔들기 angleC + 악수 v4 최종) 실행 중, GRU는 끔. 영상: 사무실 박수 → 악수 TV → 노부부 손 흔들기 → WVU 한 손 흔들기 측면 반복. 약 14.6 fps.
- 실험 동안 메모리가 부족해(가용 1.3GB, VS Code 원격 서버 2.1GB 사용) 8011 화면을 꺼 두었다가 다시 켬. 이전 실행 명령은 runs/step9_improve/original_sessions.txt에 기록.

### 이번에 만든 파일
pose_eval_v4.py(악수 비교, --stride), pose_app_v4.py(악수 v4 최종; 시도 버전은 runs/step9_hs/pose_app_v4_try1.py·try2_nose.py), train_pose_v4.py(--lowdrop, --balance), models/action_r1_ld.pt·action_r1_ldb.pt·action_r2m.pt·action_r2m_ld.pt·action_r2m_b.pt, data/action_r2/, data/action_test_v7/, media/pose/test_clips/(손 흔들기 테스트 구간 2개), media/pose/mixkit/ 악수 8개·학습용 20개, runs/step9_hs/(악수 비교·실시간 재현 스크립트와 결과, video_stream_port.py), runs/step9_loop/(round_1~5.md, r2~r5 스크립트·로그), pose_v4_video.log

## 7. 정확도 개선 2차 (2026-10-27)

6절까지의 문제: GRU가 4동작이 아닌 움직임(팔벌려뛰기, 악수, 손 들기)을 박수·손 흔들기로 자주 착각해서 실시간 화면에 켤 수 없었다.
이번에는 데이터만 늘리지 않고 "동작 아님"을 따로 배우게 하고, 앱에서 애매한 판정을 버리는 방향으로 시험했다.
모든 최종 판단은 실제 앱 재현(runs/step9_hs/live_repro.sh: 영상 하나를 2번 반복 재생하며 앱의 동작 기록을 셈)으로 했다. 라운드별 기록은 runs/step9_loop2/round_1~4.md.

평가 영상(실제 앱)
- 동작 아님: 팔벌려뛰기(jumping_jacks_burpees), 교실 손 들기(mk35968), 질문 손 들기(mk11553), 걷는 사람들(vtest), WVU 팔벌려뛰기 사람 3 정면·측면(e8_s3_v1·v3), 악수 TV(라운드 4부터)
- 팔짱: mk29963(+ 라운드 4부터 팔짱 끼는 모델 mk50643), 박수: applause_wikiconvention(+ 라운드 4부터 회의 박수 mk24214, 학습 영상이라 참고), 손 흔들기: wave_example(규칙이 맡음)
- 위 영상은 모두 GRU 학습에 쓰지 않음(mk24214만 예외).

### 라운드 요약
| 라운드 | 가설·변경 | 동작 아닌 영상의 GRU 오판정 | 맞힌 팔짱·박수 | 판단 |
| --- | --- | --- | --- | --- |
| (기준) 규칙만, GRU 끔 | - | 0 | 0 | 팔짱·박수는 못 잡음 |
| (기준) r2m 단독 (6절 최고) | - | 40 | 팔짱 3, 박수 9 | 오판정 많음 |
| 1 | "그 밖의 움직임(other)" 클래스 추가 → action_l2r1.pt (WVU 끄덕이기·주먹 지르기·조깅·차기·줍기·던지기·볼링 + Mixkit 춤·스트레칭·운동·손짓 13개) | 12 | 팔짱 3, 박수 7 | 개선 |
| 2 | 2단계: l2r1이 문지기("동작 아님"이면 버림), r2m이 동작 종류 + 판정 거부(GRU는 팔짱·박수만, 1·2위 확률 차이 0.3, 3번 연속, 규칙 확인) | 0 | 팔짱 0, 박수 3 | 개선(처음으로 오판정 0) |
| 3 | 원인 찾기: 규칙 확인을 빼면 팔짱 9·박수 6을 맞히지만 오판정 12, 연속 조건을 2로 줄여도 팔짱 0 | 0~12 | - | 개선 없음. 팔짱을 막은 건 팔짱 규칙 확인, 오판정을 막은 건 박수 규칙 확인 |
| 4 | 규칙 확인을 박수에만 적용 | **0** | **팔짱 10, 박수 3** (+학습 영상 박수 11) | **개선(최종)** |

특징 테스트(eval_gru.py)로 보면 l2r1은 팔짱 12%·박수 0%로 약하지만, other 테스트(WVU 사람 3)의 93%를 "동작 아님(other·기본)"으로 맞혀 문지기로는 쓸 만하다. r2m은 other 테스트의 50%를 박수·손 흔들기로 착각한다. 그래서 두 모델을 나눠 맡기는 2단계가 효과가 있었다.
라운드 3 이후 함께 학습해 본 action_l2r3b.pt(클래스 균형)·action_l2r3ld.pt(하체 지우기)는 특징 테스트에서 팔짱이 각각 43%·62%로 l2r1보다 나았지만 박수·손 흔들기가 0~3%라 실제 앱 평가까지는 하지 않았다.

### 최종 조합 (라운드 4)
- 앱: pose_app_v6.py (pose_app_v4.py의 손흔들기 angleC·악수 v4 규칙 + GRU 판정 거부 옵션)
- 주 모델: models/action_r2m.pt (동작 종류), 문지기: models/action_l2r1.pt (other 클래스 포함)
- 설정: --gate-th 0.5 (문지기가 idle/other라고 50% 이상이면 버림), --ml-only arms_crossed,clap (손 흔들기는 규칙), --ml-margin 0.3, --ml-vote 3 (약 1초 동안 3번 연속 같은 예측), --ml-rule --ml-rule-labels clap (박수는 최근 1.5초의 25% 이상 두 손을 붙인 '박수 치는 사람'일 때만)
- GRU가 idle/other라고 하면 규칙 판정을 덮어쓰지 않는다(이전 앱은 idle이면 '기본'으로 덮어씀).

실제 앱 재현 (영상 2번 반복)
| 영상 | 기대 | GRU 판정 |
| --- | --- | --- |
| 팔벌려뛰기, 교실 손 들기, 질문 손 들기, 걷는 사람들, WVU 팔벌려뛰기 정면·측면, 악수 TV (7개) | 동작 아님 | 모두 0 |
| mk29963 팔짱 | 팔짱 | arms_crossed 8 |
| mk50643 팔짱 끼는 모델 | 팔짱 | arms_crossed 2, clap 3 (틀림) |
| applause_wikiconvention 군중 박수 | 박수 | clap 3 |
| mk24214 회의 박수 (학습 영상) | 박수 | clap 11 |
| wave_example | 손 흔들기(규칙) | 0 (규칙 손흔들기 10) |

- 동작 아닌 영상에서 GRU 오판정 0으로 규칙만 쓸 때와 같으면서, 규칙으로는 못 잡던 팔짱·박수를 잡는다.
- 남은 문제: 팔짱을 끼고 푸는 동작(mk50643)에서 두 팔이 가슴 앞에서 움직일 때 박수로 3번 판정. 박수 규칙 확인('두 손을 붙임')을 팔짱 끼는 순간이 통과한 것으로 보임.
- 규칙 쪽 판정(만세·한팔들기)은 팔벌려뛰기·군중 박수 영상에서 여전히 많다. 이건 GRU와 별개인 규칙의 한계(5·6절).

### 실행 방법
```
cd ~/eg2-lab && source ~/eg2/bin/activate
python pose_app_v6.py --port 8011 --cam http://127.0.0.1:8096/stream \
  --model models/action_r2m.pt --gate-model models/action_l2r1.pt --gate-th 0.5 \
  --ml-only arms_crossed,clap --ml-margin 0.3 --ml-vote 3 --ml-rule --ml-rule-labels clap
```
- 웹캠이면 --cam 0, IP 카메라면 --cam http://127.0.0.1:8091/stream. 다른 포트를 쓰려면 --port만 바꾸면 된다.
- GRU 없이 규칙만: python pose_app_v6.py --port 8011 --cam ... (pose_app_v4.py와 같은 동작)
- 메모리: 앱 하나가 약 1.5GB. 다른 앱과 같이 켜면 가용 메모리를 먼저 확인할 것.

### 8011 화면
- 이번 실험 동안 메모리 때문에 8011(pose_app_v4)을 꺼 두었다. 실험 중 pose_app_v4.py가 8020(IP 카메라 8091 입력, tmux pose8020)으로 실행되고 있었고, 둘 다 켜면 가용 메모리가 300MB 안팎으로 떨어져서 사용자에게 물어봄 → 사용자 결정: 8020은 그대로 두고 8011은 켜지 않음. 새 GRU 설정은 위 실행 방법으로 필요할 때 켠다.
- 이전 실행 명령은 runs/step9_improve/original_sessions.txt에 기록.

### 이번에 만든 파일
pose_app_v5.py(GRU 판정 거부·라벨 제한·규칙 확인 옵션), pose_app_v6.py(+문지기 모델, --ml-rule-labels; 라운드 3 시점 사본 runs/step9_loop2/pose_app_v6_r3.py), models/action_l2r1.pt·action_l2r3b.pt·action_l2r3ld.pt, data/action_l2r1/(other 포함 5클래스), data/action_test_l2/(other 테스트 포함), media/pose/wvu/ 동작 2·6·7·9·10·11·12 영상(사람 1~3 × 시점 4), media/pose/mixkit/mk*_other_*.mp4 13개, runs/step9_loop2/(round_1~4.md, r1~r4 스크립트·로그·재현 결과 jsonl)
