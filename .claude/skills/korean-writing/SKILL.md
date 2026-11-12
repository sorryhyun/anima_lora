---
name: korean-writing
description: Wording rules for Korean documents written for outside readers — guidelines, guidebooks, tech reports and other full-length docs (e.g. docs/techreport/, docs/guidelines/). Load before writing or revising such a document in Korean. Not for chat replies, commit messages, code comments or internal notes.
---

# 대외용 한글 문서 작성

가이드라인, 가이드북, 기술 리포트처럼 밖에서 읽을 한글 문서를 쓰거나 고칠 때 적용합니다. 채팅 답변,
커밋 메시지, 주석, 내부 메모에는 적용하지 않습니다.

## 쓰지 않는 표현

한국인 독자에게 어색하게 들리거나, 번역투나 AI가 쓴 글처럼 읽히는 표현입니다.

| 쓰지 않음 | 대신 | 이유 |
|---|---|---|
| 꼬리 (분포의 tail) | `tail` 그대로, 또는 의미에 맞게 entropy 등 | 한국어 문서에서 분포의 tail을 "꼬리"라고 쓰면 어색합니다 |
| "A는 B 하나다", "A는 B 중 하나다" (강조 용도) | "A는 B다" | "하나"를 강조로 쓰면 외국인이나 AI가 쓴 문장으로 읽힙니다. 실제로 여러 개 중 하나라는 뜻일 때만 씁니다 |
| 조용히 (silently, without warning의 번역) | 뺍니다 | 코딩·연구 맥락에서 강조 부사로만 읽힙니다. 경고 없이 일어난다는 사실이 중요하면 "경고 없이", "오류 없이"처럼 사실을 씁니다 |
| "A가 아니라 B다" | "B다" | 아무도 A라고 하지 않았는데 A를 부정하는 문장입니다. "A가 아니라" 부분을 뺍니다 |

## 예시

- 고치기 전: "이 tail의 꼬리 부분에서 gradient가 커진다." → 고친 후: "σ 분포의 tail에서 gradient가 커진다."
- 고치기 전: "step 0은 layout을 잃는 가장 큰 단계 하나다." → 고친 후: "step 0에서 layout을 가장 많이 잃는다."
- 고치기 전: "base GPTQ를 쓰면 bake된 delta가 조용히 사라진다." → 고친 후: "base GPTQ를 쓰면 bake된 delta가 빠진다."
- 고치기 전: "레벨을 정하는 것은 noise가 아니라 objective다." → 고친 후: "레벨은 objective가 정한다."

## 확인

다 쓴 뒤 문서에서 `꼬리`, `하나`, `조용히`, `아니라`를 검색해 남은 것을 고칩니다. 실제로 수를 세는
"하나"만 둡니다.
