# Reddit → Anki vocabulary kit

Reddit 기술 글에서 고급 영어 어휘를 골라 원문 예문과 함께 Anki에 넣는 작은 자동화입니다.

## 구성

- `anki_crawler.py`, `vocab_filters.py` — Reddit RSS 수집, 어휘 필터링, AnkiConnect 카드 등록
- `eval/` — 필터 평가 스크립트와 테스트
- `automation-dashboard/` — 수집 상태·최근 단어·실행 설정을 보여주는 대시보드 모듈
- `com.minjaeku.anki-crawling.plist` — macOS launchd 예시

## 빠른 시작

```bash
cp config.example.json config.json
python3 anki_crawler.py --dry-run
python3 -m unittest discover -v
```

실제 카드 등록에는 Anki가 실행 중이고 AnkiConnect가 활성화되어 있어야 합니다. API 키와 개인 설정은 `.env` 또는 로컬 `config.json`에만 두세요.

## 자동 실행

`com.minjaeku.anki-crawling.plist`의 경로를 현재 폴더에 맞춘 뒤 `launchctl bootstrap`으로 등록합니다. 대시보드 모듈은 기존 Python 서버에서 `/api/anki`와 정적 파일 경로를 연결해 사용합니다.

생성되는 `state.sqlite3`, 로그, 일일 요약, 평가 결과는 저장소에 올리지 않습니다.
