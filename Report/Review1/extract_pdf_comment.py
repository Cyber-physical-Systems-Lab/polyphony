import fitz  # PyMuPDF
import json
import sys


def get_nearby_text(page, rect, padding=20):
    """
    Extract text near the annotation rectangle.
    Expands the rectangle slightly to capture context ("full line").
    """
    expanded_rect = fitz.Rect(
        rect.x0 - padding,
        rect.y0 - padding,
        rect.x1 + padding,
        rect.y1 + padding
    )
    text = page.get_textbox(expanded_rect)
    return text.strip()


def extract_comments(pdf_path):
    doc = fitz.open(pdf_path)

    results = []

    for page_num, page in enumerate(doc, start=1):
        annots = page.annots()

        if not annots:
            continue

        for annot in annots:
            comment = annot.info.get("content", "").strip()
            annot_type = annot.type[1]

            rect = annot.rect

            nearby_text = get_nearby_text(page, rect)

            result = {
                "page": page_num,
                "type": annot_type,
                "review_comment": comment,
                "context_text": nearby_text
            }

            results.append(result)

    return results


def save_outputs(results, base_name="comments"):
    # JSON (for LLM structured input)
    with open(f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # TXT (for quick inspection / Codex-friendly)
    with open(f"{base_name}.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[Page {r['page']}]\n")
            f.write(f"Context: {r['context_text']}\n")
            f.write(f"Review comment: {r['review_comment']}\n")
            f.write("-" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_comments.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    comments = extract_comments(pdf_path)
    save_outputs(comments)

    print(f"Extracted {len(comments)} comments.")
    print("Outputs:")
    print(" - comments.json (structured for LLM)")
    print(" - comments.txt (readable)")