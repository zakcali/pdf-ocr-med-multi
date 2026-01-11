import os
import base64
import io
import time
from multiprocessing import Pool, current_process
from pdf2image import convert_from_path
from openai import OpenAI
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_FOLDER = "pdf-in"
OUTPUT_FOLDER = "md-out"

API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "QuantTrio/Qwen3-VL-32B-Instruct-AWQ"

# This will process 8 PDF FILES simultaneously
CONCURRENCY = 8
DPI = 200 
# =================================================

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_client():
    """Create a unique client for each worker process"""
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=600)

def ocr_single_image(client, image, page_num, doc_name):
    """Sends a single image to vLLM"""
    base64_image = encode_image(image)

    prompt_text = (
        "Transcribe this medical document page into Markdown format verbatim.\n"
        "Extract text exactly as it appears.\n"
        "Represent tables using Markdown syntax (| Header |).\n"
        "Maintain strict accuracy for medical dosages and numerical values.\n"
        "Make headings bold"
        "Do not output reasoning or conversational fillers."
    )
        
    prompt_text_tr = (
        "Bu resimdeki Türkçe tıbbi belgeyi Markdown formatına çevir. "
        "Kurallar:\n"
        "1. Metni olduğu gibi, kelimesi kelimesine Türkçe olarak yaz.\n"
        "2. Tabloları mutlaka Markdown tablosu (| | |) olarak oluştur.\n"
        "3. Tıbbi terimleri ve sayıları hatasız aktar.\n"
        "4. Başlıkları kalın yap.\n"
        "5. Yorum yapma, sadece metni ver."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt_text_tr},
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.01,
            top_p=0.1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stream=False,
            extra_body={"repetition_penalty": 1.0}
        )
        content = response.choices[0].message.content
        if content:
            content = content.replace("```markdown", "").replace("```", "").strip()
        return content

    except Exception as e:
        print(f"[{doc_name}] Page {page_num} Error: {e}")
        return f"\n<!-- Page {page_num} Error -->\n"

def process_entire_document(pdf_path):
    """
    Worker function: Handles ONE PDF file from start to finish.
    """
    try:
        doc_name = os.path.basename(pdf_path)
        
        # 1. Setup Client for this worker
        client = get_client()

        # 2. Output Path
        rel_path = os.path.relpath(pdf_path, INPUT_FOLDER)
        output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(rel_path)[0] + ".md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 3. Convert PDF to Images
        # Note: This runs on CPU. With 8 workers, CPU usage might spike briefly.
        images = convert_from_path(pdf_path, dpi=DPI)
        
        full_markdown = f"# Document: {doc_name}\n\n"

        # 4. Process Pages sequentially *for this document*
        # (Because we are already running 8 documents in parallel, 
        #  we don't need to parallelize pages inside the document)
        for i, image in enumerate(images):
            page_content = ocr_single_image(client, image, i + 1, doc_name)
            full_markdown += f"\n<!-- Page {i+1} -->\n{page_content}\n"

        # 5. Save
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
        
        return f"Done: {doc_name} ({len(images)} pages)"

    except Exception as e:
        return f"Failed: {pdf_path} | {e}"

def main():
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)

    pdf_files = []
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF documents.")
    print(f"Processing 8 FILES simultaneously...")

    # Use imap to get a progress bar
    with Pool(processes=CONCURRENCY) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_entire_document, pdf_files),
            total=len(pdf_files),
            desc="Total Progress"
        ))

    # Optional: Print summary
    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()