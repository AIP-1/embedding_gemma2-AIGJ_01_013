# 8단계 결과 (2026-10-23, IP 카메라 VIGI C540-W 848x480 보조 스트림, app.py 포트 8002)
카메라: USB 웹캠은 팀원이 사용 중이라 IP 카메라 사용. LABELS에 ("yellow duck plush toy","노란 오리 인형"), ("teddy bear","곰 인형") 추가.

| 물체 | yolo 라벨 | relabel 라벨 (점수) | yoloe 라벨 | 비고 |
| --- | --- | --- | --- | --- |
| 컵 | cup (정답) | 노란 오리 인형 (0.741, 오답) | 미확인 | relabel이 맞는 라벨을 잘못 바꿈 |
| 휴대폰 | cell phone (정답) | 휴대폰 (0.699) | 미확인 | |
| 노트북 | laptop (정답) | 노트북/모니터/의자/휴대폰 섞임 | laptop (정답) | relabel 오답 많음 |
| 모니터 | tv (근접 오인식) | 모니터 | 미검출 | relabel이 개선 |
| 노란 오리 인형 (COCO 밖) | teddy bear (오인식) | 노란 오리 인형 (0.754~0.772) | yellow duck plush toy (정답) | 두 개선 모두 해결 |

| 모드 | 검출 fps | 임베딩 ms/개 | 저장 크롭 수 |
| --- | --- | --- | --- |
| yolo | 16.8 | 603 | 50 |
| relabel | 25.2 | 649 | 22 |
| yoloe | 22.0 | 619 | 15 |

정리: COCO 밖 물체(오리 인형)는 relabel·yoloe 모두 바로잡음. relabel은 LABEL_MIN 0.20이 낮아 맞는 라벨까지 바꾸는 부작용이 있어 기준을 0.7 근처로 올려야 함.
yoloe는 지정한 라벨만 찾아 오검출이 적고 속도도 yolo와 비슷해 강의실 서비스에는 yoloe가 적합.
