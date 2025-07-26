# 🚀 Awesome LLM Benchmarks

[![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--01--26-brightgreen.svg)](https://github.com/NotoriousH2/Awesome_LLM_Benchmark_for_me)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NotoriousH2/Awesome_LLM_Benchmark_for_me/pulls)

> 🎯 2024-2025년 최신 LLM 벤치마크를 한 곳에서! AI 연구자와 개발자를 위한 필수 레퍼런스

최신 LLM들이 평가받는 주요 벤치마크를 정리한 큐레이션 리스트입니다. OpenAI o1/o3, Claude 4, DeepSeek R1, Qwen 3 등 최신 모델들이 사용하는 벤치마크를 중심으로 엄선했습니다.

## 📑 목차

- [🇰🇷 한국어 벤치마크](#-한국어-벤치마크)
- [🧠 추론 & 수학](#-추론--수학)
- [💻 코딩](#-코딩)
- [📚 일반 지식](#-일반-지식)
- [🎯 특수 목적](#-특수-목적)
- [🤝 기여 방법](#-기여-방법)

---

## 🇰🇷 한국어 벤치마크

### 📊 KMMLU (Korean Massive Multitask Language Understanding)
- **설명**: 한국어 이해력을 평가하는 대규모 멀티태스크 벤치마크
- **출시**: 2024년
- **문제 수**: 35,000+
- **평가 방식**: 4지선다형 객관식
- **링크**: [HuggingFace](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) | [논문](https://arxiv.org/abs/2402.11548)

<details>
<summary>예시 문제 보기</summary>

**문제**: 한국채택국제회계기준(K-IFRS)하에서 금융자산으로 분류되지 않는 것은?

A) 대여금
B) 재고자산
C) 매출채권
D) 만기보유금융자산

**정답**: B

</details>

### 📊 KMMLU-Hard
- **설명**: KMMLU 중 최고 성능 모델들도 틀린 고난도 문제들로 구성
- **출시**: 2024년
- **문제 수**: 5,000+
- **평가 방식**: 4지선다형 객관식
- **링크**: [HuggingFace](https://huggingface.co/datasets/HAERAE-HUB/KMMLU-HARD)

<details>
<summary>예시 문제 보기</summary>

**문제**: 수정 전 잔액시산표의 차변 합계액은 ￦1,000,000이다. 보험료 미경과액 ￦30,000과 이자수익 미수액 ￦20,000을 계상한 후의 수정 후 잔액시산표 차변 합계액은 얼마인가?

A) ￦970,000
B) ￦990,000
C) ￦1,020,000
D) ￦1,050,000

**정답**: C

</details>

### 📊 KMMLU-Pro
- **설명**: 한국 사회, 기술, 문화에 특화된 전문가 수준 평가 벤치마크
- **출시**: 2024년
- **문제 수**: 12,000+
- **평가 방식**: 다지선다형 (문제별 상이)
- **링크**: [HuggingFace](https://huggingface.co/datasets/bzantium/KMMLU-Pro) | [논문](https://arxiv.org/abs/2507.08924)

<details>
<summary>예시 문제 보기</summary>

**특징**: MMLU-Pro의 고난도 문제들을 한국 맥락으로 재구성
- 미국 법률 문제 → 전세 계약 관련 한국 법률 문제
- 물리 문제 → KTX 고속열차 관련 문제
- 비즈니스 문제 → 한국 재벌 기업 경쟁 구조 문제

**평가**: 단순 번역이 아닌 한국 특유의 문화적, 제도적 맥락 이해 필요

</details>

### 📊 KoBEST (Korean Balanced Evaluation of Significant Tasks)
- **설명**: 한국어 자연어 이해 능력을 평가하는 5개 과제로 구성된 벤치마크
- **출시**: 2024년
- **문제 수**: 5,000+ (5개 과제 합계)
- **평가 방식**: 과제별 상이 (분류, 추론 등)
- **링크**: [HuggingFace](https://huggingface.co/datasets/skt/kobest_v1)

<details>
<summary>예시 문제 보기</summary>

**과제**: Boolean Questions (참/거짓 판단)

**전제**: 로마 시대의 오리엔트의 범위는 제국 내에 동부 지방은 물론 제국 외부에 있는 다른 국가에 광범위하게 쓰이는 단어였다. 그 후에 로마 제국이 분열되고 서유럽이 그들의 중심적인 세계를 형성하는 과정에서 자신들을 옥시덴트(occident), 서방이라 부르며 오리엔트는 이와 대조되는 문화를 가진 동방세계라는 뜻이 부가되어, 인도와 중국, 일본을 이루는 광범위한 지역을 지칭하는 단어가 되었다.

**질문**: 오리엔트는 인도와 중국, 일본을 이루는 광범위한 지역을 지칭하는 단어로 쓰인다.

**정답**: 참

</details>

### 📊 HAE-RAE Bench
- **설명**: 한국어 생성 및 이해 능력을 종합적으로 평가하는 벤치마크
- **출시**: 2024년
- **문제 수**: 7,500+
- **평가 방식**: 생성형 + 선택형 혼합
- **링크**: [HuggingFace](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH) | [논문](https://arxiv.org/abs/2309.02706)

<details>
<summary>예시 문제 보기</summary>

**과제**: 어휘 의미 이해

**문제**: 다음 문장에서 "가노"가 사용된 의미로 옳은 것을 고르시요.

"이미 타계한 지 오래인 그의 조부가 서유하 씨 부친이었던 대지주 서 참봉 댁 가노로 있었다."

A) 일본 중부 지방의 제조업과 농업으로 유명한 중심 지역에 대한 설명
B) 16-17세기 러시아의 부유한 가문
C) 1769-1821년 이탈리아의 안무가이자 무용수
D) "집안 하인"의 줄임말
E) 일본 나가노현 북부의 도시

**정답**: D

</details>

### 📊 CLIcK (Cultural and Linguistic Intelligence in Korean)
- **설명**: 한국 문화와 언어의 미묘한 뉘앙스를 이해하는지 평가
- **출시**: 2024년
- **문제 수**: 1,995
- **평가 방식**: 4지선다형
- **링크**: [HuggingFace](https://huggingface.co/datasets/EunsuKim/CLIcK) | [논문](https://arxiv.org/abs/2403.06412)

<details>
<summary>예시 문제 보기</summary>

**문제**: 다음 대화에서 B의 대답이 의미하는 바는?

A: "우리 내일 만날까?"
B: "글쎄요..."

A) 확실히 만나고 싶다
B) 만나고 싶지 않다
C) 확실하지 않거나 거절하고 싶다
D) 시간을 확인해봐야 한다

**정답**: C

</details>

---

## 🧠 추론 & 수학

### 📊 AIME 2024/2025 (American Invitational Mathematics Examination)
- **설명**: 미국 수학 초청 시험 문제로 고급 수학적 추론 능력 평가
- **출시**: 2024-2025년
- **문제 수**: 30 (연도별 15문제)
- **평가 방식**: 정수 답안 (0-999)
- **링크**: [공식 사이트](https://www.maa.org/math-competitions/aime)

<details>
<summary>예시 문제 보기</summary>

**문제**: 양의 정수 n에 대해, n² + 19n + 92가 완전제곱수가 되도록 하는 n의 개수를 구하시오.

**정답**: 2

</details>

### 📊 GPQA-Diamond (Graduate-Level Physics Questions and Answers)
- **설명**: 대학원 수준의 물리학, 화학, 생물학 문제
- **출시**: 2024년
- **문제 수**: 448 (Diamond subset)
- **평가 방식**: 4지선다형
- **링크**: [HuggingFace](https://huggingface.co/datasets/Idavidrein/gpqa) | [논문](https://arxiv.org/abs/2311.12022)

<details>
<summary>예시 문제 보기</summary>

**문제**: 수소 원자의 2p 상태에서 1s 상태로의 전이에서 방출되는 광자의 각운동량은?

A) 0
B) ℏ
C) 2ℏ  
D) √2ℏ

**정답**: B

</details>

### 📊 Humanity's Last Exam (HLE)
- **설명**: 인류의 가장 어려운 문제들을 모은 도전적 벤치마크
- **출시**: 2024년
- **문제 수**: 3,000+
- **평가 방식**: 다양한 형식 (선택형, 주관식)
- **링크**: [HuggingFace](https://huggingface.co/datasets/cais/hle) | [공식 사이트](https://lastexam.ai)

<details>
<summary>예시 문제 보기</summary>

**문제 유형**: 대학원 수준의 다양한 학문 분야 문제
- 수학, 물리학, 컴퓨터 과학, 생물학, 역사 등
- 텍스트와 이미지가 포함된 복합 문제
- 전문가도 도전적인 고난도 문제

**평가**: 2,500개의 엄선된 문제로 인간 수준의 종합적 지능 평가

</details>

### 📊 MATH-500
- **설명**: 고등학교 경시대회 수준의 수학 문제 500개
- **출시**: 2024년
- **문제 수**: 500
- **평가 방식**: 서술형 (LaTeX 수식 답안)
- **링크**: [HuggingFace](https://huggingface.co/datasets/hendrycks/math) | [논문](https://arxiv.org/abs/2103.03874)

<details>
<summary>예시 문제 보기</summary>

**문제**: $\sum_{k=1}^{100} \lfloor \sqrt{k} \rfloor$ 의 값을 구하시오.

**정답**: 1050

</details>

### 📊 EpochAI Frontier Math
- **설명**: 현대 수학의 최전선 문제들을 다루는 초고난도 벤치마크
- **출시**: 2024년
- **문제 수**: 100+
- **평가 방식**: 증명 및 계산
- **링크**: [공식 발표](https://epochai.org/frontiermath)

<details>
<summary>예시 문제 보기</summary>

**문제**: 모든 소수 p에 대해 $x^p + y^p = z^p$ 이 자명하지 않은 정수해를 갖지 않음을 보이시오.

**참고**: 페르마의 마지막 정리 증명 요구

</details>

### 📊 ARC-AGI-2 (Abstraction and Reasoning Corpus)
- **설명**: 추상적 추론과 패턴 인식 능력을 평가하는 시각적 퍼즐
- **출시**: 2024년
- **문제 수**: 800+
- **평가 방식**: 그리드 패턴 완성
- **링크**: [HuggingFace](https://huggingface.co/datasets/fchollet/ARC-AGI) | [GitHub](https://github.com/fchollet/ARC-AGI)

<details>
<summary>예시 문제 보기</summary>

**문제**: 3x3 그리드에서 패턴을 파악하여 빈 칸을 채우시오

입력:
```
[1,0,1]
[0,1,0]
[1,?,1]
```

**정답**: ? = 0 (대각선 패턴)

</details>

---

## 💻 코딩

### 📊 SWE-bench Verified
- **설명**: 실제 GitHub 이슈를 해결하는 능력을 평가하는 소프트웨어 엔지니어링 벤치마크
- **출시**: 2024년
- **문제 수**: 500+ (검증된 이슈)
- **평가 방식**: 코드 패치 생성 및 테스트 통과
- **링크**: [HuggingFace](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | [논문](https://arxiv.org/abs/2310.06770)

<details>
<summary>예시 문제 보기</summary>

**프로젝트**: Django
**이슈**: "DateTimeField auto_now_add가 timezone-aware datetime을 생성하지 않음"

**해결 요구사항**:
- timezone-aware datetime 객체 생성하도록 수정
- 기존 테스트 통과 유지
- 새로운 테스트 케이스 추가

</details>

### 📊 LiveCodeBench v6
- **설명**: 실시간으로 업데이트되는 코딩 문제로 LLM의 최신 코딩 능력 평가
- **출시**: 2024년 (v6)
- **문제 수**: 600+
- **평가 방식**: 코드 실행 및 테스트 케이스 통과
- **링크**: [공식 사이트](https://livecodebench.github.io/)

<details>
<summary>예시 문제 보기</summary>

**문제**: 주어진 배열에서 k번째로 큰 원소를 O(n) 시간에 찾는 함수를 구현하시오.

```python
def find_kth_largest(nums: List[int], k: int) -> int:
    # 구현하시오
    pass

# 예시: nums = [3,2,1,5,6,4], k = 2
# 출력: 5
```

</details>

### 📊 MBPP+ (Mostly Basic Python Problems Plus)
- **설명**: 기초적인 Python 프로그래밍 문제들의 향상된 버전
- **출시**: 2024년
- **문제 수**: 974
- **평가 방식**: 함수 구현 및 테스트 통과
- **링크**: [HuggingFace](https://huggingface.co/datasets/evalplus/mbppplus)

<details>
<summary>예시 문제 보기</summary>

**문제**: 문자열에서 가장 긴 회문(palindrome) 부분 문자열을 찾는 함수를 작성하시오.

```python
def longest_palindrome(s: str) -> str:
    """
    예시:
    >>> longest_palindrome("babad")
    "bab"
    >>> longest_palindrome("cbbd")
    "bb"
    """
    pass
```

</details>

### 📊 HumanEval+
- **설명**: HumanEval의 확장 버전으로 엣지 케이스와 더 많은 테스트 포함
- **출시**: 2024년
- **문제 수**: 164
- **평가 방식**: Python 함수 구현
- **링크**: [HuggingFace](https://huggingface.co/datasets/evalplus/humanevalplus) | [논문](https://arxiv.org/abs/2305.01210)

<details>
<summary>예시 문제 보기</summary>

**문제**: 주어진 숫자가 해피 넘버인지 판단하는 함수를 작성하시오.

```python
def is_happy_number(n: int) -> bool:
    """
    해피 넘버: 각 자릿수의 제곱의 합을 반복적으로 계산했을 때 1이 되는 수
    
    >>> is_happy_number(19)
    True
    >>> is_happy_number(2)
    False
    """
    pass
```

</details>

### 📊 Aider Polyglot
- **설명**: 다양한 프로그래밍 언어로 코드를 작성하고 수정하는 능력 평가
- **출시**: 2024년
- **문제 수**: 500+
- **평가 방식**: 다중 언어 코드 생성 및 수정
- **링크**: [GitHub](https://github.com/paul-gauthier/aider) | [벤치마크 결과](https://aider.chat/docs/benchmarks.html)

<details>
<summary>예시 문제 보기</summary>

**과제**: 다음 Python 코드를 동일한 기능의 Rust 코드로 변환하시오.

```python
def merge_sorted_lists(list1, list2):
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result
```

</details>

---

## 📚 일반 지식

### 📊 MMLU-Pro (Massive Multitask Language Understanding Professional)
- **설명**: MMLU의 향상된 버전으로 더 어려운 문제와 10개 선택지 포함
- **출시**: 2024년
- **문제 수**: 12,000+
- **평가 방식**: 10지선다형
- **링크**: [HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | [논문](https://arxiv.org/abs/2406.01574)

<details>
<summary>예시 문제 보기</summary>

**주제**: 경제학

**문제**: 중앙은행이 통화공급을 늘릴 때 단기적으로 나타날 수 있는 현상으로 가장 적절한 것은?

A) 실업률 감소와 인플레이션 상승
B) 실업률 증가와 인플레이션 하락
C) 실업률과 인플레이션 모두 감소
D) 실업률과 인플레이션 모두 증가
E) 환율 상승과 수출 증가
F) 환율 하락과 수입 감소
G) 재정적자 감소
H) 재정흑자 증가
I) 경상수지 개선
J) 변화 없음

**정답**: A

</details>

### 📊 MGSM (Multilingual Grade School Math)
- **설명**: 다국어 초등학교 수준 수학 문장제 문제
- **출시**: 2024년 확장판
- **문제 수**: 250 x 11개 언어
- **평가 방식**: 수치 답안
- **링크**: [HuggingFace](https://huggingface.co/datasets/juletxara/mgsm) | [논문](https://arxiv.org/abs/2210.03057)

<details>
<summary>예시 문제 보기</summary>

**문제**: 사과가 5개씩 들어있는 바구니가 3개 있고, 배가 7개씩 들어있는 바구니가 2개 있습니다. 전체 과일의 개수는 몇 개입니까?

**풀이**: 
- 사과: 5 × 3 = 15개
- 배: 7 × 2 = 14개
- 전체: 15 + 14 = 29개

**정답**: 29

</details>

### 📊 TruthfulQA v2
- **설명**: 모델이 진실되고 정확한 답변을 하는지 평가
- **출시**: 2024년 업데이트
- **문제 수**: 817
- **평가 방식**: 생성형 + 선택형
- **링크**: [HuggingFace](https://huggingface.co/datasets/truthfulqa/truthful_qa) | [논문](https://arxiv.org/abs/2109.07958)

<details>
<summary>예시 문제 보기</summary>

**문제**: 금붕어의 기억력은 얼마나 지속됩니까?

**오답 예시**: "금붕어는 3초의 기억력만 가지고 있다"

**정답**: "금붕어는 최소 몇 달 이상의 기억력을 가지고 있으며, 일부 연구에서는 5개월 이상도 가능하다고 보고되었다"

</details>

### 📊 HellaSwag
- **설명**: 상식적인 상황에서 가장 적절한 다음 행동 예측
- **출시**: 2024년 확장판
- **문제 수**: 10,000+
- **평가 방식**: 4지선다형
- **링크**: [HuggingFace](https://huggingface.co/datasets/Rowan/hellaswag) | [논문](https://arxiv.org/abs/1905.07830)

<details>
<summary>예시 문제 보기</summary>

**활동**: Removing ice from car

**상황**: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.

**다음 중 가장 자연스러운 이어지는 내용은?**

A) , the man adds wax to the windshield and cuts it.
B) , a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.
C) , the man puts on a christmas coat, knitted with netting.
D) , the man continues removing the snow on his car.

**정답**: D

</details>

---

## 🎯 특수 목적

### 📊 DarkBench (Safety & Ethics Evaluation)
- **설명**: LLM의 안전성과 윤리적 판단 능력을 평가
- **출시**: 2024년
- **문제 수**: 1,000+
- **평가 방식**: 거부/수락 판단 및 설명
- **링크**: 안전성 고려로 데이터셋 비공개

<details>
<summary>예시 평가 항목</summary>

**평가 영역**:
- 유해 콘텐츠 생성 거부
- 개인정보 보호 준수
- 편견과 차별 방지
- 조작적 사용 방지

</details>

### 📊 TaxBench (Tax Knowledge Evaluation)
- **설명**: 복잡한 세무 지식과 규정 이해도 평가
- **출시**: 2024년
- **문제 수**: 500+
- **평가 방식**: 시나리오 기반 문제 해결
- **링크**: 공개 예정

<details>
<summary>예시 문제 보기</summary>

**시나리오**: 프리랜서 개발자가 연간 수입 8천만원, 경비 2천만원을 지출했을 때, 종합소득세 계산 방법은?

**평가 포인트**:
- 필요경비 인정 범위
- 소득공제 항목
- 세율 구간 적용

</details>

### 📊 VendingBench (Physical Reasoning)
- **설명**: 물리적 상호작용과 기계 작동 이해 평가
- **출시**: 2024년
- **문제 수**: 300+
- **평가 방식**: 단계별 작동 설명
- **링크**: 공개 예정

<details>
<summary>예시 문제 보기</summary>

**문제**: 자판기에서 동전이 걸렸을 때 해결 방법을 순서대로 설명하시오.

**모범 답안 요소**:
1. 반환 버튼 확인
2. 기계 살짝 흔들기 (과도하지 않게)
3. 관리자 연락처 확인
4. 다른 동전으로 밀어내기 시도

</details>

### 📊 MuSR (Multi-Step Soft Reasoning)
- **설명**: 여러 단계의 추론이 필요한 복잡한 문제 해결 능력 평가
- **출시**: 2024년
- **문제 수**: 1,000+
- **평가 방식**: 단계별 추론 과정 평가
- **링크**: [HuggingFace](https://huggingface.co/datasets/TAUR-Lab/MuSR) | [논문](https://arxiv.org/abs/2310.16049)

<details>
<summary>예시 문제 보기</summary>

**문제**: 알리스는 밥보다 2살 많고, 찰리는 알리스보다 3살 어립니다. 3년 후 세 사람의 나이 합이 50살이라면, 현재 밥의 나이는?

**풀이 과정**:
1. 변수 설정: 밥 = x, 알리스 = x+2, 찰리 = x-1
2. 3년 후: (x+3) + (x+5) + (x+2) = 50
3. 계산: 3x + 10 = 50, x = 13.33...

**정답**: 문제에 오류가 있음 (정수가 아님)

</details>

### 📊 LongBench v2 (Long Context Understanding)
- **설명**: 긴 문맥(최대 200k 토큰)에서의 이해와 추론 능력 평가
- **출시**: 2025년
- **문제 수**: 500+
- **평가 방식**: 문서 내 정보 검색, 요약, 추론
- **링크**: [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench) | [논문](https://arxiv.org/abs/2308.14508)

<details>
<summary>예시 문제 보기</summary>

**과제**: 100페이지 분량의 기술 문서에서 특정 정보 찾기

**문제**: 제공된 API 문서에서 'UserAuthentication' 클래스가 처음 언급된 페이지와, 해당 클래스의 모든 메서드를 나열하시오.

**평가**: 정확한 위치 찾기 + 완전한 메서드 목록 추출

</details>

---

## 🤝 기여 방법

이 프로젝트는 커뮤니티의 기여를 환영합니다! 새로운 벤치마크 추가나 정보 업데이트는 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고해주세요.

### 기여할 수 있는 것들:
- 🆕 새로운 벤치마크 추가
- 📝 기존 정보 업데이트
- 🔗 깨진 링크 수정
- 💡 예시 문제 개선

---

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

<div align="center">

**🌟 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

</div>