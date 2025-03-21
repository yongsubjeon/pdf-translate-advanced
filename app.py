from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import tempfile
import logging
import json
import base64
import time
import asyncio
import numpy as np
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2
import re
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

# .env 파일에서 환경 변수 로드
load_dotenv()

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 업로드 폴더가 없으면 생성
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 로그 디렉토리 생성
if not os.path.exists('logs'):
    os.makedirs('logs')

# 로깅 설정
logger = logging.getLogger('translate_app')
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# 파일 핸들러 추가 (10MB 크기 제한, 백업 5개)
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
file_handler.setFormatter(file_format)

# 핸들러 등록
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 기본 로거 설정 방지 (이미 설정했으므로)
logger.propagate = False

# Upstage API 설정
UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY', '')  # .env 파일에서 가져온 API 키

# LLM API 설정 (번역용)
TRANSLATION_API_URL = os.getenv('TRANSLATION_API_URL', 'https://api.upstage.ai/v1/chat/completions')
# Document OCR API 설정
DOCUMENT_OCR_API_URL = os.getenv('DOCUMENT_OCR_API_URL', 'https://api.upstage.ai/v1/document-ai/document-parse')
# Embedding API 설정
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', 'https://api.upstage.ai/v1/solar/embeddings')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/translate-pdf', methods=['POST'])
def translate_pdf():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 제공되지 않았습니다.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'PDF 파일만 지원합니다.'}), 400
    
    try:
        # 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        file.save(temp_file.name)
        temp_file.close()
        
        # 처리 방식 선택 (OCR vs PyPDF2)
        use_ocr = True  # 레이아웃 보존을 위해 OCR 사용
        
        if use_ocr:
            # Upstage Document OCR API로 PDF 내용 추출 (레이아웃 보존)
            extracted_text, document_structure = extract_text_with_ocr(temp_file.name)
        else:
            # PyPDF2로 PDF 내용 추출 (원본 레이아웃 유지 시도)
            extracted_text = extract_text_from_pdf(temp_file.name)
            document_structure = None
        
        # 추출된 텍스트가 없거나 오류 메시지인 경우
        if not extracted_text or len(extracted_text.strip()) == 0 or "오류가 발생했습니다" in extracted_text:
            translated_text = "PDF에서 텍스트를 추출할 수 없습니다."
        else:
            # 텍스트가 너무 길면 청크로 나누어 병렬 번역
            translated_text = translate_with_structure(extracted_text, document_structure)
        
        # 임시 파일 삭제
        os.unlink(temp_file.name)
        
        return jsonify({'translation': translated_text})
    
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return jsonify({'error': f'PDF 처리 중 오류가 발생했습니다: {str(e)}'}), 500

def extract_text_with_ocr(file_path):
    """Upstage Document OCR API를 사용하여 PDF 분석 (레이아웃 보존)"""
    try:
        # 요청 설정
        headers = {
            "Authorization": f"Bearer {UPSTAGE_API_KEY}"
        }
        
        # multipart/form-data 형식으로 파일 전송
        with open(file_path, "rb") as f:
            files = {
                "document": (os.path.basename(file_path), f, "application/pdf")
            }
            
            # 추가 매개변수 설정
            data = {
                "ocr": "force",              # OCR 강제 실행
                "coordinates": "true",        # 각 요소의 좌표 반환
                "chart_recognition": "true",  # 차트 인식 활성화
                "output_formats": '["html"]', # HTML 형식으로 출력
                "model": "document-parse"     # 문서 파싱 모델 사용
            }
            
            # API 요청
            response = requests.post(
                DOCUMENT_OCR_API_URL, 
                headers=headers, 
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                data = response.json()
                logging.info(f"OCR API 성공적으로 처리됨: {len(data.get('elements', []))} 요소 발견")
                
                # HTML 구조와 문서 정보 함께 저장 (이미지 포함)
                document_structure = {
                    'html_content': data.get('content', {}).get('html', ''),
                    'elements': data.get('elements', []),
                    'pages': []
                }
                
                # 요소 분석 및 페이지별 구조화
                pages_count = 0
                for elem in data.get('elements', []):
                    page_num = elem.get('page', 1)
                    pages_count = max(pages_count, page_num)
                
                # 추출된 요소로부터 구조화된 텍스트 생성 (이미지 포함)
                extracted_text = generate_structured_text_from_elements(data.get('elements', []), pages_count)
                
                return extracted_text, document_structure
            else:
                logging.error(f"Document OCR API error: 상태 코드 {response.status_code}, 응답: {response.text}")
                # 실패 시 PyPDF2 대체 사용
                fallback_text = extract_text_from_pdf(file_path)
                return fallback_text, None
    except Exception as e:
        logging.error(f"OCR 처리 중 오류 발생: {str(e)}")
        # 예외 발생 시 PyPDF2 대체 사용
        fallback_text = extract_text_from_pdf(file_path)
        return fallback_text, None

def format_table(table_data):
    """표 데이터를 HTML 테이블로 포맷팅"""
    if not table_data or not isinstance(table_data, list) or len(table_data) == 0:
        return ""
    
    # HTML 테이블 형식으로 변환
    table_html = "<table border='1' cellpadding='3' cellspacing='0' style='width:100%; border-collapse:collapse;'>\n"
    
    # 첫 번째 행을 헤더로 처리
    table_html += "  <thead>\n    <tr>\n"
    for cell in table_data[0]:
        cell_content = cell if isinstance(cell, str) else str(cell)
        table_html += f"      <th style='background-color:#f2f2f2; text-align:left; padding:6px;'>{cell_content}</th>\n"
    table_html += "    </tr>\n  </thead>\n"
    
    # 나머지 행을 처리
    table_html += "  <tbody>\n"
    for row in table_data[1:]:
        table_html += "    <tr>\n"
        for cell in row:
            cell_content = cell if isinstance(cell, str) else str(cell)
            table_html += f"      <td style='text-align:left; padding:4px; border:1px solid #ddd;'>{cell_content}</td>\n"
        table_html += "    </tr>\n"
    table_html += "  </tbody>\n"
    
    table_html += "</table>"
    return table_html

def generate_structured_text_from_elements(elements, total_pages):
    """OCR 요소로부터 구조화된 텍스트 생성 (레이아웃 및 요소 유형 보존)"""
    if not elements:
        return ""
    
    # 페이지별로 요소 그룹화
    page_elements = {}
    for elem in elements:
        page = elem.get('page', 1)
        if page not in page_elements:
            page_elements[page] = []
        page_elements[page].append(elem)
    
    # 각 페이지별로 요소를 y 좌표 기준으로 정렬하여 처리
    result_text = ""
    for page_num in sorted(page_elements.keys()):
        # 페이지 헤더 추가
        result_text += f"\n[페이지 {page_num}/{total_pages}]\n\n"
        
        # 요소를 y 좌표 기준으로 정렬 (위에서 아래로)
        sorted_elements = sorted(
            page_elements[page_num], 
            key=lambda e: e.get('coordinates', [{}])[0].get('y', 0) 
            if e.get('coordinates') and len(e.get('coordinates', [])) > 0 else 0
        )
        
        # 각 요소를 처리
        for elem in sorted_elements:
            category = elem.get('category', '')
            elem_id = elem.get('id', '')
            content = ""
            
            # 콘텐츠 추출
            if 'content' in elem:
                if 'text' in elem['content'] and elem['content']['text']:
                    content = elem['content']['text']
                elif 'html' in elem['content'] and elem['content']['html']:
                    # HTML에서 텍스트 추출 (간단한 방법)
                    html_content = elem['content']['html']
                    # 기본적인 HTML 태그 제거
                    content = re.sub(r'<[^>]+>', ' ', html_content)
                    content = re.sub(r'\s+', ' ', content).strip()
            
            # 요소 유형별 처리 (이미지/표 특별 처리)
            if category == 'heading1' or category.startswith('header'):
                result_text += f"<제목>{content}</제목>\n\n"
            elif category == 'heading2':
                result_text += f"<부제목>{content}</부제목>\n\n"
            elif category == 'paragraph':
                result_text += f"{content}\n\n"
            elif category == 'list':
                result_text += f"<목록>\n{content}\n</목록>\n\n"
            elif category == 'table':
                # 테이블 처리 개선 - 테이블 내용과 페이지 번호 및 좌표 추가
                table_content = ""
                if 'table' in elem:
                    table_content = format_table(elem.get('table', []))
                
                # 좌표 정보 추가 (이미지 추출 용도)
                coords = ""
                if 'coordinates' in elem and len(elem['coordinates']) >= 2:
                    top_left = elem['coordinates'][0]
                    bottom_right = elem['coordinates'][2] if len(elem['coordinates']) > 2 else elem['coordinates'][1]
                    coords = f"data-coord=\"top-left:({int(top_left['x']*1000)},{int(top_left['y']*1000)}); bottom-right:({int(bottom_right['x']*1000)},{int(bottom_right['y']*1000)})\""
                
                result_text += f"<표 id='{elem_id}' page='{page_num}' {coords}>\n{table_content}\n</표>\n\n"
            elif category == 'image' or category == 'figure':
                # 이미지 정보 추가 (좌표 정보 포함)
                image_desc = content or '이미지'
                coords = ""
                if 'coordinates' in elem and len(elem['coordinates']) >= 2:
                    top_left = elem['coordinates'][0]
                    bottom_right = elem['coordinates'][2] if len(elem['coordinates']) > 2 else elem['coordinates'][1]
                    coords = f"data-coord=\"top-left:({int(top_left['x']*1000)},{int(top_left['y']*1000)}); bottom-right:({int(bottom_right['x']*1000)},{int(bottom_right['y']*1000)})\""
                
                result_text += f"<이미지 id='{elem_id}' page='{page_num}' {coords}>{image_desc}</이미지>\n\n"
            elif category == 'chart':
                # 차트 정보 추가 (좌표 정보 포함)
                chart_desc = content or '차트'
                chart_data = ""
                if 'table' in elem:
                    chart_data = format_table(elem.get('table', []))
                
                coords = ""
                if 'coordinates' in elem and len(elem['coordinates']) >= 2:
                    top_left = elem['coordinates'][0]
                    bottom_right = elem['coordinates'][2] if len(elem['coordinates']) > 2 else elem['coordinates'][1]
                    coords = f"data-coord=\"top-left:({int(top_left['x']*1000)},{int(top_left['y']*1000)}); bottom-right:({int(bottom_right['x']*1000)},{int(bottom_right['y']*1000)})\""
                
                result_text += f"<차트 id='{elem_id}' page='{page_num}' {coords}>{chart_desc}\n{chart_data}</차트>\n\n"
            elif category == 'footer':
                result_text += f"<푸터>{content}</푸터>\n\n"
            else:
                # 기타 요소
                result_text += f"{content}\n\n"
    
    return result_text

def extract_document_structure(ocr_data):
    """OCR 응답에서 문서 구조 정보 추출"""
    structure = {
        'pages': []
    }
    
    if 'pages' in ocr_data:
        for page in ocr_data['pages']:
            page_structure = {
                'blocks': [],
                'tables': []
            }
            
            # 블록 정보 추출
            if 'blocks' in page:
                for block in page['blocks']:
                    block_info = {
                        'type': block.get('type', 'text'),
                        'bbox': block.get('bbox', []),
                        'text': block.get('text', '')
                    }
                    
                    # 표 정보 추출
                    if 'table' in block and block['table']:
                        table_info = {
                            'rows': len(block['table']),
                            'cols': len(block['table'][0]) if block['table'] else 0,
                            'content': block['table']
                        }
                        page_structure['tables'].append(table_info)
                    
                    page_structure['blocks'].append(block_info)
            
            structure['pages'].append(page_structure)
    
    return structure

def extract_text_from_pdf(file_path):
    """PyPDF2를 사용하여 PDF에서 텍스트 추출"""
    try:
        pages_text = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                # 원문 텍스트 추출 (줄바꿈 유지)
                page_text = page.extract_text()
                
                if page_text:
                    # 페이지 번호 추가
                    page_header = f"\n[페이지 {page_num+1}/{total_pages}]\n"
                    pages_text.append(page_header + page_text)
        
        # 모든 페이지 텍스트 합치기 (줄바꿈 유지)
        all_text = "\n\n".join(pages_text)
        return all_text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return f"PDF 텍스트 추출 중 오류가 발생했습니다: {str(e)}"

def clean_text(text):
    """추출된 텍스트 정리 - 원본 레이아웃을 최대한 유지하는 방식으로 수정"""
    # 불필요한 제어 문자만 제거 (줄바꿈과 공백은 유지)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

def split_text_into_chunks(text, max_length=3800):
    """텍스트를 번역을 위한 청크로 분할"""
    # 청크 분할 시작 로그
    logger.info(f"텍스트 청크 분할 시작: 총 길이 {len(text)} 문자")
    
    chunks = []
    current_chunk = ""
    
    # 페이지 단위로 먼저 분할
    pages = re.split(r'(\n\[페이지 \d+/\d+\]\n)', text)
    logger.info(f"페이지 기준 분할: {len(pages)} 섹션")
    
    i = 0
    while i < len(pages):
        # 페이지 헤더가 있는 경우
        if i < len(pages) - 1 and re.match(r'\n\[페이지 \d+/\d+\]\n', pages[i]):
            header = pages[i]
            content = pages[i+1] if i+1 < len(pages) else ""
            logger.debug(f"페이지 헤더 처리: {header.strip()}, 내용 길이: {len(content)} 문자")
            
            # 페이지 내용이 최대 길이보다 길면 단락 단위로 분할
            if len(content) > max_length:
                paragraphs = re.split(r'(\n\n+)', content)
                logger.debug(f"단락 분할: {len(paragraphs)} 단락")
                temp_chunk = header
                
                for para in paragraphs:
                    if len(temp_chunk) + len(para) <= max_length:
                        temp_chunk += para
                    else:
                        if temp_chunk != header:  # 헤더만 있는 경우는 제외
                            chunks.append(temp_chunk)
                            logger.debug(f"청크 추가: 길이 {len(temp_chunk)} 문자")
                        temp_chunk = header + para
                
                if temp_chunk != header:  # 마지막 청크 추가
                    chunks.append(temp_chunk)
                    logger.debug(f"마지막 청크 추가: 길이 {len(temp_chunk)} 문자")
            else:
                chunks.append(header + content)
                logger.debug(f"페이지 전체를 하나의 청크로 추가: 길이 {len(header + content)} 문자")
            
            i += 2  # 헤더와 내용을 함께 처리했으므로 2 증가
        else:
            # 헤더 없는 일반 내용
            content = pages[i]
            if current_chunk and len(current_chunk) + len(content) <= max_length:
                current_chunk += content
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    logger.debug(f"일반 청크 추가: 길이 {len(current_chunk)} 문자")
                
                if len(content) > max_length:
                    # 큰 청크는 단락 단위로 분할
                    paragraphs = re.split(r'(\n\n+)', content)
                    logger.debug(f"큰 내용 단락 분할: {len(paragraphs)} 단락")
                    current_chunk = ""
                    
                    for para in paragraphs:
                        if len(current_chunk) + len(para) <= max_length:
                            current_chunk += para
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                                logger.debug(f"분할된 청크 추가: 길이 {len(current_chunk)} 문자")
                            current_chunk = para
                else:
                    current_chunk = content
            
            i += 1
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
        logger.debug(f"최종 청크 추가: 길이 {len(current_chunk)} 문자")
    
    logger.info(f"청크 분할 완료: 총 {len(chunks)} 청크")
    # 각 청크 크기 확인 로그
    for i, chunk in enumerate(chunks):
        logger.debug(f"청크 #{i+1}: {len(chunk)} 문자")
    
    return chunks

def translate_with_structure(text, document_structure=None):
    """텍스트를 구조 정보를 유지하며 번역"""
    try:
        logger.info(f"번역 시작: 전체 텍스트 길이 {len(text)} 문자")
        
        # 텍스트를 적절한 크기로 분할
        chunks = split_text_into_chunks(text)
        
        # 병렬 번역을 위한 ThreadPoolExecutor 사용
        logger.info(f"병렬 번역 시작: {len(chunks)} 청크")
        translated_chunks = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 모든 청크를 병렬로 번역 (개별 결과 추적)
            future_to_chunk = {executor.submit(translate_text, chunk): i for i, chunk in enumerate(chunks)}
            for future in future_to_chunk:
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    translated_chunks.append(result)
                    logger.info(f"청크 #{chunk_index+1} 번역 완료: 원본 {len(chunks[chunk_index])} → 번역 {len(result)} 문자")
                except Exception as e:
                    logger.error(f"청크 #{chunk_index+1} 번역 실패: {str(e)}")
                    # 실패한 청크는 원본 텍스트로 대체
                    translated_chunks.append(f"[번역 실패: {str(e)}]\n{chunks[chunk_index]}")
        
        # 번역된 청크 결합
        translated_text = "".join(translated_chunks)
        logger.info(f"번역 완료: 최종 번역 길이 {len(translated_text)} 문자")
        
        # 구조 정보가 있는 경우 특수 처리 (표, 차트 등)
        if document_structure:
            # 표 형식 보존 처리 등 추가 작업 가능
            logger.debug("문서 구조 정보로 추가 처리 (사용되지 않음)")
        
        return translated_text
    except Exception as e:
        logger.error(f"Error in translate_with_structure: {str(e)}")
        # 실패 시 일반 번역 시도
        logger.info("전체 번역 실패, 일부 텍스트만 번역 시도")
        return translate_text(text[:4000]) + "\n\n(번역 중 오류가 발생하여 일부만 번역되었습니다.)"

def get_embedding(text, model="embedding-query"):
    """텍스트의 임베딩 벡터 생성"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {UPSTAGE_API_KEY}"
        }
        
        payload = {
            "model": model,
            "input": text
        }
        
        response = requests.post(EMBEDDING_API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            embedding = data.get('data', [{}])[0].get('embedding', [])
            return embedding
        else:
            logging.error(f"Embedding API error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error getting embedding: {str(e)}")
        return None

def translate_text(text):
    """LLM API를 사용하여 영어 텍스트를 한글로 번역"""
    try:
        # 빈 텍스트는 번역하지 않음
        if not text or len(text.strip()) == 0:
            logger.debug("빈 텍스트 건너뜀")
            return ""
        
        # 토큰 수 추정 (영문 기준 약 4자에 1토큰)
        estimated_tokens = len(text) // 4
        logger.info(f"번역 요청: 텍스트 길이 {len(text)} 문자, 약 {estimated_tokens} 토큰 (추정)")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {UPSTAGE_API_KEY}"
        }
        
        # 태그와 구조를 보존하는 번역 프롬프트
        prompt = """당신은 영어를 한국어로 번역하는 전문 번역가입니다. 다음 지침을 엄격히 따라주세요:

1. 영어 텍스트를 정확하고 자연스러운 한국어로 번역하세요.
2. 아래 특수 태그는 번역하지 말고 그대로 유지하세요:
   - <제목></제목>, <부제목></부제목>
   - <표></표>, <이미지></이미지>, <차트></차트>
   - <목록></목록>, <푸터></푸터>
   - [페이지 X/Y] 형식의 페이지 번호
3. 태그 내부의 영어 콘텐츠만 한국어로 번역하세요.
4. 표 형식과 구조는 유지하며 내용만 번역하세요.
5. 숫자, 단위, 통계 데이터는 그대로 유지하세요.
6. 전문 용어는 적절한 한국어 용어로 번역하되, 필요시 괄호 안에 원문을 유지하세요.
7. 'USD Million', 'CAGR' 등의 금융/시장 용어는 일반적인 한국어 표현을 사용하세요.

번역할 텍스트:"""
        
        # Upstage LLM API 요청 본문
        payload = {
            "model": "solar-1-mini-chat",  # 사용할 모델명
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.1,  # 번역의 일관성을 위해 낮은 temperature 설정
            "max_tokens": 4096   # 충분한 출력 토큰 확보
        }
        
        # 요청 시간 기록
        start_time = time.time()
        
        # 3번까지 재시도
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.debug(f"API 요청 시도 #{retry_count+1}")
                response = requests.post(TRANSLATION_API_URL, json=payload, headers=headers)
                
                # 응답 시간 기록
                response_time = time.time() - start_time
                logger.info(f"API 응답 수신: 상태 코드 {response.status_code}, 응답 시간 {response_time:.2f}초")
                
                if response.status_code == 200:
                    data = response.json()
                    translated_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    # 토큰 사용량 확인 (있을 경우)
                    if 'usage' in data:
                        prompt_tokens = data['usage'].get('prompt_tokens', 0)
                        completion_tokens = data['usage'].get('completion_tokens', 0)
                        total_tokens = data['usage'].get('total_tokens', 0)
                        logger.info(f"토큰 사용량: 입력 {prompt_tokens}, 출력 {completion_tokens}, 총 {total_tokens}")
                    
                    logger.info(f"번역 완료: 결과 길이 {len(translated_text)} 문자")
                    return translated_text
                elif response.status_code == 429:  # 속도 제한
                    retry_count += 1
                    wait_time = 2 * retry_count
                    logger.warning(f"속도 제한 (429): {wait_time}초 대기 후 재시도 #{retry_count}")
                    time.sleep(wait_time)  # 지수 백오프
                else:
                    logger.error(f"Translation API error: 상태 코드 {response.status_code}, 응답 내용: {response.text}")
                    return f"번역 중 오류가 발생했습니다 (상태 코드: {response.status_code})\n원인: {response.text}"
            except requests.exceptions.RequestException as e:
                retry_count += 1
                wait_time = 2 * retry_count
                logger.error(f"API 요청 실패 (시도 #{retry_count}): {str(e)}, {wait_time}초 대기")
                time.sleep(wait_time)
        
        logger.error(f"최대 재시도 횟수 ({max_retries}회) 초과: 번역 실패")
        return "번역 서버 연결 실패 (여러 번 재시도 후)"
    except Exception as e:
        logger.error(f"번역 처리 중 예외 발생: {str(e)}")
        return f"번역 중 오류가 발생했습니다: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True) 