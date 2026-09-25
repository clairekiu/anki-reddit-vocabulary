# anki-crawling

Reddit 기술 글(RSS) → DeepSeek로 B2~C2 어휘 + 원문 예문 추출 → AnkiConnect로 `English::reddit-vocab` 덱에 추가.

- 매일 20:30 launchd(`com.minjaeku.anki-crawling`) 실행, 하루 10~15개, `state.sqlite3`로 단어/글 중복 방지
- Anki가 꺼져 있으면 자동 실행 시도, 실패 시 카드는 보류(pending)되어 다음 실행에 재전송, 추가 후 AnkiWeb sync
- 결과는 `daily_summary.json` → 자동화 대시보드(`/anki`) 및 21시 브리핑(`evening_briefing.py --anki`)
- `disabled` 파일이 있으면 건너뜀. 설정은 `config.json`(없으면 기본값)

```bash
/usr/bin/python3 anki_crawler.py --dry-run   # 수집만
/usr/bin/python3 -m unittest -v
```

## 논문 모드

```bash
/usr/bin/python3 anki_crawler.py --paper paper.pdf            # 또는 arXiv/PDF URL, .txt
/usr/bin/python3 anki_crawler.py --paper URL --deck "English::..." --max 10
```

같은 필터·중복 DB를 쓰고 기본 덱은 `English::paper-vocab`(최대 15개), 결과는 JSON. 텔레그램에서 hermes-yuna에게
PDF·링크를 보내며 "안키로 정리해줘"라고 하면 `skills/anki-connect` 스킬이 이 명령을 실행한다. PDF 추출에 `poppler`(brew) 필요.

## 필터 (vocab_filters.py)

일일 수집·논문 모드·평가가 같은 `screen()`을 쓴다. 순서: 스키마 → 비영어(표제어 비ASCII, 예문 언어 판별) → 코드/도구명
→ CEFR(C1+, 여러 단어 표현은 B2+) → 기초 표현 블록리스트(어간 매칭) → 예문 원문 검증 → 단어 포함 확인 →
LLM 판정(여러 단어 표현·사전에 없는 단어: 관용/비자명 여부, 일반 명사구, 코드명) → 중복(동일형·Porter 어간·표현 계열) → 글당 상한.
사전은 `/usr/share/dict/words` + 보강 목록.

## 평가

```bash
/usr/bin/python3 eval/run_eval.py            # 임시 DB + English::eval-test 덱(끝나면 삭제), --no-anki 가능
```

같은 LLM 후보에 이전 필터와 새 필터를 모두 적용해 `eval/results/compare_*.csv`로 전후 비교한다.
