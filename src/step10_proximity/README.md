# 10단계 카메라 2대 강의실 접근 감지 결과 공유 (정지원 → 팀원, 2026-10-28)

읽기 전용 복사본이에요. 필요한 파일은 자기 폴더로 복사해서 쓰세요 (`cp -r /var/tmp/eg2-share-jiwon934/step10 ~/eg2-lab/share_from_jiwon934_step10`).
원본은 `/home/jiwon934/eg2-lab/prox/`. 사람이 찍힌 이미지와 모델 가중치(*.pt, *.ts)는 넣지 않았어요.

## 핵심 결과 (자세한 건 runs/step10_results.md)
- 점 위에 서서 잰 타겟까지 거리 오차 평균: ankle1(발목, 카메라 1대) -22cm / stereo(어깨 삼각측량) -13cm / fused -10cm. 위치 오차 평균 32 / 22 / 24cm. 셋 다 거리를 짧게 잼.
- stereo가 ankle1보다 약 10cm 정확. 발목은 의자·책상에 가리면 못 잡음.
- fused의 single(한 카메라 추정, 어깨 1.2m 가정)은 위치가 50–90cm 틀리고, 책상에 앉은 사람을 통로의 유령 사람으로 만들어 잘못된 근접 알람을 냄 → proximity_app_cpu.py에서 single은 알람에서 뺌.
- 10-6 타겟 "red mickey mouse trash can": (0.07, 0.07, 0.11) 삼각측량, 재투영 0.9px. 시나리오 다가가기·1m 10초 서 있기·2m 지나가기·앉기는 맞음, 이탈(3m 밖) 알람은 안 남 (멀어지면 cam2 시야 밖 → single → 알람 제외).
- 성능 (CPU): ankle1 2.3fps, stereo·fused 1.2fps, 카메라 시간 차 1–25ms, 타겟 모드 RAM 3.1GB.
- **GPU를 못 썼어요**: /dev/nvgpu/igpu0에 장치가 없고 `torch.cuda.is_available()` = False ("NvRmGpuLibOpen failed"). 그래서 모든 결과가 CPU 기준이고 fps가 낮아요. 원본 proximity_app.py의 타겟 모드는 EmbeddingGemma를 cuda로 고정해서 GPU가 없으면 죽어요 → proximity_app_cpu.py를 쓰세요.

## 좌표계
- 원점 O = 기둥과 벽이 만나는 안쪽 모서리 바닥 (빨간 미키 휴지통 자리).
- x = 벽에서 수직으로 통로 쪽(+), y = 벽을 따라 모서리에서 멀어지는 쪽(+), z = 위(+), 단위 m.
- 바닥 흰 점: P1 (0.62, 0.32), P2 (0.62, 0.77), P5 (1.24, 0.32), P4 (1.24, 0.77). → calib/world_points.json
- 앱은 기준점 범위 ±0.5m 밖(x -0.5~3.5, y -1.0~4.5) 사람을 버려요.

## 카메라 배치 (카메라를 움직이면 calib을 다시 해야 해요)
| 이름 | 장치 | 보정 파일 | 위치 (m) | 비고 |
|---|---|---|---|---|
| cam0 | /dev/video0 | calib/cam0_intr.npz, cam0.npz | (0.77, 3.12, 1.57) | 통로 쪽에서 모서리를 봄, 재투영 평균 1.06px |
| cam2 | /dev/video2 | calib/cam2_intr.npz, **cam1.npz** | (0.30, 2.44, 2.32), 아래로 37.5° | 10월 28일 14:15에 옮기고 다시 보정, 평균 2.67px |

- 외부 보정 파일 이름은 앱 순서(0, 1)라 cam2(video2)의 위치는 cam1.npz예요.
- 두 카메라 교차 검증 (삼각측량 vs 실제): 0.4–2.4cm.
- cam2를 처음엔 1.53m 높이에 두었는데, 점 위에 선 사람의 어깨가 화면 위로 벗어나 stereo가 하나도 안 잡혔어요. 서 있는 사람의 어깨(1.4m)와 바닥 점이 같이 들어오는지 꼭 확인하세요.
- 카메라가 움직였는지 확인: 지금 화면의 점 위치를 calib/clicks.json과 비교 (1–2px 이내면 그대로 써도 됨).

## 실행 방법
venv는 각자 것을 쓰고, **포트는 자기 번호**를 쓰세요 (8020은 제가 쓰던 번호). 아래 예시는 8030.

```bash
cd ~/eg2-lab/share_from_jiwon934_step10      # 복사한 폴더
# 1) 초점 고정 (앱 켜기 전에. 실행 중에 바꾸면 I/O 오류). cam2 보정은 focus_absolute=19 상태에서 했어요
for d in 0 2; do v4l2-ctl -d /dev/video$d -c focus_automatic_continuous=0 -c focus_absolute=19; done
# 2) 모드별 실행 (브라우저 http://<Jetson IP>:8030)
python proximity_app_cpu.py --cams 0 2 --mode ankle1 --target-xy 0,0 --port 8030   # 10-3
python proximity_app_cpu.py --cams 0 2 --mode stereo --target-xy 0,0 --port 8030   # 10-4
python proximity_app_cpu.py --cams 0 2 --mode fused  --target-xy 0,0 --port 8030   # 10-5
python proximity_app_cpu.py --cams 0 2 --target "red mickey mouse trash can" --near 1.0 --far 3.0 --port 8030   # 10-6
python proximity_app_cpu.py --cams 0 2 --target "..." --moving --port 8030          # 10-8 (안 해봄)
# 3) 기록 (선택): state_logger.py 안의 URL 포트를 8030으로 바꾼 뒤
python -u state_logger.py     # → runs/<mode>_log.csv, 0.3초마다
python summarize_log.py runs/fused_log.csv <시작 epoch초>   # 트랙별 최소 거리·알람·점프 요약
```

- 필요한 모델: yolo11n-pose.pt, yoloe-11s-seg.pt (ultralytics가 처음에 받음), 타겟 모드는 mobileclip_blt.ts(572MB)도 필요 → `~/eg2-lab/` 등에 있으면 같은 폴더에 링크해 두세요 (`ln -s ~/eg2-lab/mobileclip_blt.ts .`). 없으면 느리게 다시 받아요.
- 카메라는 한 프로그램만 열 수 있어요. 끌 때 Ctrl+C가 안 먹으면 `kill -9 <PID>` 후 몇 초 기다렸다 다시 켜야 "카메라 0을 열 수 없음"이 안 나요.
- 보정 다시 하기: `calib_intrinsics_live.py --cam 0 --port 8030`(렌즈, 체커보드 9x6, 2.5cm), `calib_extrinsics.py --cams 0 2 --port 8030`(브라우저에서 기준점 클릭 → cam0.npz, cam1.npz).
- 알람 소리: 페이지의 "알람 소리 켜기"를 눌러야 나요. Safari/Chrome에서 안 나서 맥에서 mac_alarm.py로 소리를 냈어요 (/state를 보고 새 '알람'마다 afplay 3번; 파일 안 URL을 Jetson IP와 자기 포트로).

## 파일
| 파일 | 내용 |
|---|---|
| proximity_app.py | 원본 앱 (ankle1 / stereo / fused, 타겟 텍스트 검색, --moving) — 오늘 수정 안 함 |
| proximity_app_cpu.py | 원본 사본 + 수정: GPU 없으면 CPU, 페이지 소리 Safari 대응(<audio> + 빨간 화면 깜빡임), single은 알람 제외, 같은 자리 0.5m·5초 안 근접 알람 생략, 트랙 유지 2→4초 |
| proximity_app_lower.py | stereo 매칭을 하반신(발목→무릎→엉덩이)으로 바꾼 판. 과제가 어깨 기준이라 안 씀, 실행 안 해봄 |
| stereo.py | 공통: 보정 로드, 왜곡 보정, 삼각측량, 높이 평면 투영, 칼만 필터 |
| calib_intrinsics.py, calib_intrinsics_live.py | 10-1 렌즈 보정 |
| calib_extrinsics.py | 10-2 카메라 위치 보정 (기준점 클릭) |
| cam_view.py | 두 카메라 화면만 보기 |
| state_logger.py | /state를 0.3초마다 CSV로 기록 |
| summarize_log.py | 로그 CSV를 트랙별로 요약 |
| mac_alarm.py | 맥에서 알람 소리 내기 (맥에서 실행) |
| calib/*.npz, world_points.json, clicks.json | 보정 결과·기준점 좌표·클릭 위치 |
| runs/step10_results.md | 10-1~10-7 전체 정리 |
| runs/ankle1_dots.md, stereo_dots.md, fused_dots.md | 모드별 점 위 측정 표 (stereo_dots_oldcam2.md = cam2 옮기기 전, 검출 0) |
| runs/target_scenarios.md | 10-6 시나리오 결과 |
| runs/*.csv | 기록 원본 (ankle1_log, stereo_log, fused_log_round = fused 점 측정, fused_log = 10-6, *_oldcam2 = cam2 옮기기 전) |

## 2026-10-29 추가: 10-8 이동 타겟 (GPU 복구 후)
- GPU가 다시 잡혀(재부팅) 10.0–10.3fps로 실행 (10-3~10-7은 CPU 1.2–2.4fps).
- 1차(휴지통을 의자로 밀며 이동): 충돌 위험 알람 5건, 타겟 재식별 11회 — 사람이 가리면 타겟을 놓쳐 놓침이 생김.
- 2차(휴지통 고정, 한 사람): 충돌 4/5 맞춤 (놓친 1회는 접근 0.45m/s < 기준 0.5m/s), 지나가기 헛알람 0/5, 알람 때 TTC 1.0–1.5초.
- 자세한 표: runs/10-8_results.md, 회차 시각: runs/10-8_trials.csv, 기록: runs/fused_moving_log.csv
- 실행: python proximity_app_cpu.py --cams 2 0 --target "red mickey mouse trash can" --moving --port <자기 포트>
- ⚠️ 재부팅 뒤 카메라 번호가 바뀜: 지금은 USB 포트 2.4(cam0, 통로 쪽) = /dev/video2, 포트 2.2(cam2, 모서리 쪽) = /dev/video0 → `--cams 2 0`. 번호 대신 /dev/v4l/by-path/...usb-0:2.4... / ...usb-0:2.2... 로 확인하세요.
