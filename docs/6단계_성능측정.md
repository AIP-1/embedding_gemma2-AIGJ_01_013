# 6단계 결과표 (Jetson Orin Nano Super 8GB, 2026-10-23)

환경: torch 2.14.1+cu132, transformers 5.20.1, sentence-transformers 6.2.0, bf16, HF_HUB_OFFLINE=1,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (4단계 이후), batch_size=2 (4단계 이후).
주의: RAM은 tegrastats로 본 보드 전체 사용량 최고치다. 팀원 프로세스(약 3.3GB)가 함께 돌고 있었고 스왑을 4.7~5.7GB 쓰는 상태였다. 모델 단독 사용량이 아니다.

| 구성 | 보드 RAM 최고(MB) | 입력 | 지연 ms (2회차) | GPU 최고 % | 비고 |
|---|---|---|---|---|---|
| 텍스트 전용 | 6462 | 질의 2개 (문서 4개는 미리 인코딩) | 158~191 | 97 | dim 768/256/128 모두 정답 |
| +비전 (이미지) | 7091 | 이미지 6장 | 6132 (1회차) | 99 | 같은 강의실 사진이라 점수 차 작음 |
| +비전 (동영상, 프레임 평균) | 7164 | clip.mp4 → 1초당 1프레임 21장 | 17670 | 99 | batch_size 기본값은 CUDA OOM, 2로 성공 |
| +비전 (동영상, {"video": ...}) | - | clip.mp4 | 실패 | - | torchcodec 필요, 0.17은 시스템 ffmpeg 8과 로드 실패 |
| +오디오 | 7214 | wav 3개 (16kHz 모노) | 5102 | 99 | 경보음·소음은 맞음, espeak 합성 음성 질의는 noise로 틀림 |
| 전체 (비전+오디오) | - | - | - | - | 메모리 부족 우려로 실행 안 함 |
