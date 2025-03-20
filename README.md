# PDF 번역기 (PDF Translator)

PDF 문서를 업로드하면 원본 레이아웃과 구조를 유지하면서 텍스트를 번역해주는 웹 애플리케이션입니다.

## 주요 기능

- **원본 레이아웃 보존**: 번역 시 PDF의 원본 레이아웃, 이미지, 표 구조를 최대한 유지합니다.
- **구조화된 번역**: 제목, 부제목, 단락, 표, 이미지, 차트 등 문서 요소의 구조를 인식하고 보존합니다.
- **실시간 미리보기**: PDF 원본과 번역 결과를 나란히 비교할 수 있습니다.
- **이미지 및 차트 처리**: PDF에 포함된 이미지와 차트를 추출하여 번역 결과에 포함합니다.

## 기술 스택

- **Frontend**: HTML, CSS, JavaScript, PDF.js
- **Backend**: Python, Flask
- **PDF 처리**: PDF.js, PyPDF2
- **OCR 및 문서 구조 인식**: Upstage Document Parser API
- **번역**: Upstage 번역 API

## 설치 방법

### 요구 사항

- Python 3.8 이상
- pip (Python 패키지 관리자)

### 설치 단계

1. 저장소 클론
   ```bash
   git clone https://github.com/yongsubjeon/pdf-translate-advanced.git
   cd pdf-translate-advanced
   ```

2. 가상 환경 생성 및 활성화 (선택 사항)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수 설정
   - `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.
   - Upstage API 키를 `.env` 파일에 입력합니다.
   ```
   UPSTAGE_API_KEY=your_api_key_here
   ```

## 실행 방법

```bash
python app.py
```

서버가 시작되면 `http://127.0.0.1:5000`에서 애플리케이션에 접속할 수 있습니다.

## 사용 방법

1. 웹 브라우저에서 `http://127.0.0.1:5000`에 접속합니다.
2. "파일 선택" 버튼을 클릭하여 번역할 PDF 파일을 업로드합니다.
3. 업로드가 완료되면 자동으로 번역 처리가 시작됩니다.
4. 왼쪽에는 원본 PDF가, 오른쪽에는 번역 결과가, 표시됩니다.
5. 페이지 컨트롤을 사용하여 PDF의 다른 페이지를 볼 수 있습니다.

## 애플리케이션 구조

- `app.py`: Flask 서버와 PDF 처리 및 번역 로직이 포함된 메인 파일
- `index.html`: 웹 인터페이스
- `requirements.txt`: 필요한 Python 패키지 목록
- `.env.example`: 환경 변수 템플릿

## 주요 기능 설명

### PDF 텍스트 추출

PDF에서 텍스트를 추출하는 두 가지 방법을 지원합니다:
1. PyPDF2를 사용한 기본 텍스트 추출
2. Upstage Document Parser API를 사용한 OCR 및 고급 문서 구조 인식

### 이미지 및 차트 처리

PDF.js를 사용하여 PDF에서 이미지와 차트를 추출합니다. 추출된 요소는 좌표 정보를 기반으로 원본 위치에 표시됩니다.

### 구조 보존 번역

문서의 구조적 요소(제목, 부제목, 표, 이미지 등)를 유지하면서 텍스트만 번역합니다. 특수 태그 시스템을 사용하여 번역 과정에서 구조 정보를 보존합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 개발자

PDF 번역기는 문서 번역 작업을 더 효율적으로 만들기 위해 개발되었습니다. 피드백과 기여를 환영합니다. 