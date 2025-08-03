import pytesseract
from pdf2image import convert_from_path
import os
import numpy as np
from PIL import Image
import cv2
import math
import tempfile
from pathlib import Path
import re
from chatgpt_text_enhancer import ChatGPTTextEnhancer
from dotenv import load_dotenv
from openai import OpenAI
import requests

def enhance_extracted_text_with_chatgpt(text: str, text_type: str = "general", 
                                       target_format: str = "clean_text",
                                       use_chunking: bool = True) -> tuple:
    """
    Migliora il testo estratto dall'OCR usando ChatGPT
    
    Args:
        text: Testo da migliorare
        text_type: Tipo di testo ("main", "notes", "general")
        target_format: Formato desiderato
        use_chunking: Se usare la divisione in chunk per testi lunghi
        
    Returns:
        tuple: (testo_migliorato, analisi_miglioramenti)
    """
    if not text or not text.strip():
        return text, {"status": "empty"}
    
    try:
        enhancer = ChatGPTTextEnhancer()
        
        if use_chunking and len(text) > 3000:
            return enhancer.enhance_text_in_chunks(text, text_type=text_type, target_format=target_format)
        else:
            return enhancer.enhance_text(text, text_type=text_type, target_format=target_format)
            
    except Exception as e:
        print(f"  ❌ Errore nell'enhancement con ChatGPT: {str(e)}")
        return text, {"status": "error", "error": str(e)}

def process_image_with_chatgpt(image, page_num, output_dir=None, auto_orient=True, preprocess=True, 
                              remove_fingers=True, remove_watermarks=True, correct_lighting=True,
                              correct_color=True, enhance_quality=True, save_processed_images=False,
                              enhance_contrast=True, conservative_finger_removal=True,
                              remove_noise=True, separate_notes=True, use_chatgpt=True,
                              chatgpt_format="clean_text"):
    """
    Versione migliorata della funzione process_image con integrazione ChatGPT
    """
    print(f"Processing page {page_num}...")
    
    # [Tutto il codice di preprocessing rimane uguale al tuo script originale]
    # Per brevità, assumo che il preprocessing sia già fatto e abbiamo extracted_text
    
    # Simulo il processo OCR (usa il tuo codice originale qui)
    processed_image = image  # Placeholder - usa il tuo preprocessing
    
    # Estrazione testo OCR (usa il tuo codice originale)
    try:
        configs = [
            '--oem 1 --psm 3',
            '--oem 1 --psm 6',
            '--oem 1 --psm 4',
            '--oem 1 --psm 3 -l ita',
        ]
        
        best_text = ""
        best_word_count = 0
        
        for config in configs:
            page_text = pytesseract.image_to_string(processed_image, config=config)
            word_count = len([w for w in page_text.split() if len(w) > 1])
            
            if word_count > best_word_count:
                best_text = page_text
                best_word_count = word_count
        
        extracted_text = best_text
        print(f"  Estratte {best_word_count} parole dalla pagina {page_num}")
        
        # Separazione testo principale e note (usa il tuo codice originale)
        if separate_notes:
            main_text, notes_text = separate_main_text_and_notes(processed_image, extracted_text)
        else:
            main_text = extracted_text
            notes_text = ""
        
        # NUOVA FUNZIONALITÀ: Miglioramento con ChatGPT
        if use_chatgpt:
            print(f"  🤖 Migliorando testo con ChatGPT...")
            
            # Migliora il testo principale
            if main_text.strip():
                enhanced_main, main_analysis = enhance_extracted_text_with_chatgpt(
                    main_text, "main", chatgpt_format
                )
                print(f"  ✅ Testo principale migliorato: {main_analysis.get('changes', 0)} modifiche")
                main_text = enhanced_main
            
            # Migliora le note se presenti
            if notes_text.strip():
                enhanced_notes, notes_analysis = enhance_extracted_text_with_chatgpt(
                    notes_text, "notes", chatgpt_format
                )
                print(f"  ✅ Note migliorate: {notes_analysis.get('changes', 0)} modifiche")
                notes_text = enhanced_notes
        
        return processed_image, main_text, notes_text
        
    except Exception as e:
        print(f"  ❌ Errore nell'elaborazione della pagina {page_num}: {str(e)}")
        return processed_image, f"[ERRORE: {str(e)}]", ""

def extract_text_from_scanned_pdf_with_chatgpt(pdf_path, output_txt_path=None, 
                                              output_main_txt_path=None, output_notes_txt_path=None,
                                              auto_orient=True, preprocess=True, remove_fingers=True, 
                                              remove_watermarks=True, correct_lighting=True, 
                                              correct_color=True, enhance_quality=True, 
                                              save_processed_images=False, output_dir=None, 
                                              enhance_contrast=True, conservative_finger_removal=True, 
                                              skip_incomplete_pages=True, split_double_pages=True, 
                                              remove_noise=True, separate_notes=True,
                                              use_chatgpt=True, chatgpt_format="clean_text"):
    """
    Versione migliorata della funzione principale con integrazione ChatGPT
    """
    print(f"🚀 Elaborazione PDF con ChatGPT: {pdf_path}")
    
    # Verifica che l'API key sia configurata se ChatGPT è abilitato
    if use_chatgpt:
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            print("⚠️  ATTENZIONE: OPENAI_API_KEY non trovata. ChatGPT enhancement disabilitato.")
            use_chatgpt = False
        else:
            print("✅ ChatGPT enhancement abilitato")
    
    # Crea directory per immagini processate se necessario
    if save_processed_images:
        if output_dir is None:
            output_dir = os.path.splitext(pdf_path)[0] + "_processed_images"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Immagini processate salvate in: {output_dir}")
    
    # Converti PDF in immagini
    print("📄 Conversione PDF in immagini...")
    images = convert_from_path(pdf_path)
    print(f"PDF convertito in {len(images)} immagini")
    
    # Estrai testo da ogni immagine
    print("🔍 Estrazione testo con OCR (potrebbe richiedere tempo)...")
    extracted_text = ""
    main_text_all = ""
    notes_text_all = ""
    
    page_counter = 1
    total_chatgpt_improvements = 0
    
    for i, image in enumerate(images):
        print(f"\n📖 Elaborazione immagine {i+1}/{len(images)}...")
        
        # [Tutto il codice di gestione pagine doppie e incomplete rimane uguale]
        # Per brevità, processo direttamente l'immagine
        
        # Elabora la pagina con ChatGPT
        _, page_main_text, page_notes_text = process_image_with_chatgpt(
            image, page_counter, output_dir, auto_orient, preprocess, remove_fingers, 
            remove_watermarks, correct_lighting, correct_color, enhance_quality,
            save_processed_images, enhance_contrast, conservative_finger_removal,
            remove_noise, separate_notes, use_chatgpt, chatgpt_format
        )
        
        # Combina testo principale e note per il testo completo
        page_full_text = page_main_text
        if page_notes_text:
            page_full_text += "\n\n--- NOTE ---\n\n" + page_notes_text
        
        extracted_text += f"\n\n--- PAGINA {page_counter} ---\n\n"
        extracted_text += page_full_text
        
        main_text_all += f"\n\n--- PAGINA {page_counter} ---\n\n"
        main_text_all += page_main_text
        
        notes_text_all += f"\n\n--- PAGINA {page_counter} NOTE ---\n\n"
        notes_text_all += page_notes_text
        
        page_counter += 1
    
    # Salva i file di output
    if output_txt_path:
        print(f"💾 Salvataggio testo estratto in: {output_txt_path}")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
    
    if output_main_txt_path:
        print(f"💾 Salvataggio testo principale in: {output_main_txt_path}")
        with open(output_main_txt_path, 'w', encoding='utf-8') as f:
            f.write(main_text_all)
    
    if output_notes_txt_path:
        print(f"💾 Salvataggio note in: {output_notes_txt_path}")
        with open(output_notes_txt_path, 'w', encoding='utf-8') as f:
            f.write(notes_text_all)
    
    print("🎉 Estrazione testo completata!")
    
    if use_chatgpt:
        print(f"🤖 ChatGPT ha migliorato il testo su {page_counter-1} pagine")
    
    return extracted_text, main_text_all, notes_text_all

def detect_page_completeness(image, threshold=0.3):
    """
    Determina se una pagina è completa o tagliata a metà.
    
    Args:
        image: PIL Image object
        threshold: Soglia per determinare se una pagina è incompleta
        
    Returns:
        bool: True se la pagina è completa, False se è tagliata
        float: Percentuale di completezza stimata (0.0-1.0)
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate horizontal and vertical projections
    h_proj = np.sum(thresh, axis=1)
    v_proj = np.sum(thresh, axis=0)
    
    # Normalize projections
    h_proj = h_proj / np.max(h_proj) if np.max(h_proj) > 0 else h_proj
    v_proj = v_proj / np.max(v_proj) if np.max(v_proj) > 0 else v_proj
    
    # Calculate the density of text in different regions of the page
    height, width = thresh.shape
    
    # Divide the page into quarters horizontally and vertically
    h_quarters = [np.mean(h_proj[i*height//4:(i+1)*height//4]) for i in range(4)]
    v_quarters = [np.mean(v_proj[i*width//4:(i+1)*width//4]) for i in range(4)]
    
    # Calculate the standard deviation of text density across quarters
    h_std = np.std(h_quarters)
    v_std = np.std(v_quarters)
    
    # Calculate the overall text density
    overall_density = np.mean(thresh) / 255.0
    
    # If the page has very low text density, it might be blank or severely cut off
    if overall_density < 0.01:
        return False, overall_density
    
    # If the standard deviation of text density is very high, the page might be cut off
    # (text would be concentrated in one part of the page)
    completeness_score = 1.0 - (h_std + v_std) / 2.0
    
    return completeness_score > threshold, completeness_score

def detect_double_page(image):
    """
    Determina se un'immagine contiene due pagine affiancate e le separa.
    
    Args:
        image: PIL Image object
        
    Returns:
        list: Lista di PIL Image objects (1 o 2 pagine)
        bool: True se sono state rilevate due pagine, False altrimenti
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img_cv.shape[:2]
    
    # If the width-to-height ratio is greater than 1.5, it might be a double page
    if width / height > 1.5:
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate vertical projection
        v_proj = np.sum(thresh, axis=0)
        
        # Normalize projection
        v_proj = v_proj / np.max(v_proj) if np.max(v_proj) > 0 else v_proj
        
        # Smooth the projection with a moving average filter
        window_size = width // 20
        v_proj_smooth = np.convolve(v_proj, np.ones(window_size)/window_size, mode='same')
        
        # Find the potential split point (minimum in the middle region)
        middle_region_start = width // 3
        middle_region_end = 2 * width // 3
        middle_region = v_proj_smooth[middle_region_start:middle_region_end]
        
        # If there's a clear minimum in the middle region, it's likely a double page
        if len(middle_region) > 0:
            min_idx = np.argmin(middle_region) + middle_region_start
            min_value = v_proj_smooth[min_idx]
            
            # Calculate the average value in the left and right halves
            left_avg = np.mean(v_proj_smooth[:width//2])
            right_avg = np.mean(v_proj_smooth[width//2:])
            
            # If the minimum is significantly lower than the average, it's likely a split point
            if min_value < 0.3 * (left_avg + right_avg) / 2:
                # Split the image at the detected point
                left_page = img_cv[:, :min_idx]
                right_page = img_cv[:, min_idx:]
                
                # Convert back to PIL Images
                left_pil = Image.fromarray(cv2.cvtColor(left_page, cv2.COLOR_BGR2RGB))
                right_pil = Image.fromarray(cv2.cvtColor(right_page, cv2.COLOR_BGR2RGB))
                
                # Check if both pages have enough content
                left_complete, left_score = detect_page_completeness(left_pil)
                right_complete, right_score = detect_page_completeness(right_pil)
                
                # If both pages are reasonably complete, return them
                if left_complete and right_complete:
                    return [left_pil, right_pil], True
    
    # If no double page was detected or the split wasn't successful
    return [image], False

def correct_color_cast(image):
    """
    Corregge dominanti di colore come le colorazioni bluastre ai lati.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Image with color cast corrected
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Split the image into channels
    b, g, r = cv2.split(img_cv)
    
    # Calculate the average of each channel
    r_avg = np.mean(r)
    g_avg = np.mean(g)
    b_avg = np.mean(b)
    
    # Calculate the overall average
    avg = (r_avg + g_avg + b_avg) / 3
    
    # Calculate scaling factors
    alpha_r = avg / r_avg if r_avg > 0 else 1
    alpha_g = avg / g_avg if g_avg > 0 else 1
    alpha_b = avg / b_avg if b_avg > 0 else 1
    
    # Apply color correction only if there's a significant color cast
    # Check if any channel is significantly different from the average
    if max(abs(r_avg - avg), abs(g_avg - avg), abs(b_avg - avg)) > 10:
        print("  Detected color cast, applying correction")
        
        # Apply scaling factors to each channel
        r = cv2.addWeighted(r, alpha_r, 0, 0, 0)
        g = cv2.addWeighted(g, alpha_g, 0, 0, 0)
        b = cv2.addWeighted(b, alpha_b, 0, 0, 0)
        
        # Merge the channels back
        balanced = cv2.merge([b, g, r])
        
        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
    else:
        return image

def correct_uneven_lighting(image):
    """
    Corregge illuminazione non uniforme e ombreggiature.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Image with corrected lighting
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply a large Gaussian blur to estimate the background illumination
    blur_size = max(gray.shape[0], gray.shape[1]) // 10
    if blur_size % 2 == 0:
        blur_size += 1  # Ensure odd kernel size
    
    background = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Divide the original image by the background to correct illumination
    # First normalize background to 0-1 range
    background_norm = background / 255.0
    
    # Ensure no division by zero
    background_norm = np.maximum(background_norm, 0.01)
    
    # Normalize and correct each channel
    r, g, b = cv2.split(img_cv)
    
    r_corrected = np.minimum((r / background_norm), 255).astype(np.uint8)
    g_corrected = np.minimum((g / background_norm), 255).astype(np.uint8)
    b_corrected = np.minimum((b / background_norm), 255).astype(np.uint8)
    
    # Merge channels
    corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))

def detect_fingers(image, conservative=True):
    """
    Detect and remove fingers or other obstructions in scanned images.
    
    Args:
        image: PIL Image object
        conservative: If True, uses a more conservative approach to avoid removing content
        
    Returns:
        PIL Image: Image with fingers/obstructions removed
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img_cv.shape[:2]
    
    # Convert to HSV color space for better skin detection
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Define multiple ranges for skin color in HSV to handle different lighting conditions
    # These ranges are expanded to handle various skin tones and lighting conditions
    skin_ranges = [
        # Standard skin tone range
        (np.array([0, 20, 70], dtype=np.uint8), np.array([20, 150, 255], dtype=np.uint8)),
        # Darker skin tones
        (np.array([0, 10, 60], dtype=np.uint8), np.array([20, 150, 255], dtype=np.uint8)),
        # More saturated skin tones (for reddish lighting)
        (np.array([0, 30, 70], dtype=np.uint8), np.array([25, 170, 255], dtype=np.uint8))
    ]
    
    # Create a combined mask for all skin color ranges
    combined_skin_mask = np.zeros((height, width), dtype=np.uint8)
    
    for lower_skin, upper_skin in skin_ranges:
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        combined_skin_mask = cv2.bitwise_or(combined_skin_mask, skin_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(combined_skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to identify potential fingers
    min_area = img_cv.shape[0] * img_cv.shape[1] * 0.005  # Minimum area threshold (0.5% of image)
    max_area = img_cv.shape[0] * img_cv.shape[1] * 0.15   # Maximum area threshold (15% of image)
    
    # Create a mask for the detected fingers
    finger_mask = np.zeros_like(skin_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Calculate aspect ratio of the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Check if the contour is at the edge of the image (where fingers typically appear)
            # In conservative mode, only consider the very edges of the image
            edge_threshold = 0.1 if conservative else 0.2
            
            is_at_edge = (
                x < width * edge_threshold or 
                x + w > width * (1 - edge_threshold) or 
                y < height * edge_threshold or 
                y + h > height * (1 - edge_threshold)
            )
            
            if is_at_edge:
                # Fingers typically have a small aspect ratio (height > width)
                if aspect_ratio < 1.0:
                    cv2.drawContours(finger_mask, [contour], 0, 255, -1)
    
    # Dilate the finger mask to ensure complete coverage
    finger_mask = cv2.dilate(finger_mask, kernel, iterations=2)
    
    # Invert the finger mask to get the background
    background_mask = cv2.bitwise_not(finger_mask)
    
    # Create a white background for the finger areas
    white_bg = np.ones_like(img_cv) * 255
    
    # Combine the original image (where there are no fingers) with the white background (where fingers were detected)
    result = cv2.bitwise_and(img_cv, img_cv, mask=background_mask)
    white_part = cv2.bitwise_and(white_bg, white_bg, mask=finger_mask)
    result = cv2.add(result, white_part)
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def remove_watermark(image, watermark_text="Scanned with CamScanner"):
    """
    Remove watermark from the image by detecting and replacing it with background color.
    
    Args:
        image: PIL Image object
        watermark_text: Text of the watermark to remove
        
    Returns:
        PIL Image: Image with watermark removed
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img_cv.shape[:2]
    
    # Check the bottom right corner where CamScanner watermark typically appears
    bottom_right = img_cv[height-100:height, width-300:width]
    
    # Convert to grayscale
    gray_br = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to isolate the watermark
    _, thresh_br = cv2.threshold(gray_br, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours in the bottom right corner
    contours_br, _ = cv2.findContours(thresh_br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found in the bottom right, likely a watermark
    if contours_br:
        # Fill the bottom right corner with white (or the dominant color)
        img_cv[height-100:height, width-300:width] = [255, 255, 255]
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def detect_orientation_robust(image):
    """
    A more robust method to detect text orientation using multiple approaches.
    
    Args:
        image: PIL Image object
        
    Returns:
        int: Rotation angle (0, 90, 180, or 270)
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Method 1: Try Tesseract's OSD
    try:
        osd = pytesseract.image_to_osd(img_cv)
        angle = int(osd.split('\nRotate: ')[1].split('\n')[0])
        print("  Orientation detected using Tesseract OSD")
        return angle
    except Exception as e:
        print(f"  Tesseract OSD failed: {str(e)}")
    
    # Method 2: Try OCR at different orientations and pick the one with highest confidence
    orientations = [0, 180]  # Only consider 0 and 180 degrees to avoid incorrect 90/270 rotations
    max_confidence = -1
    best_orientation = 0
    
    for angle in orientations:
        # Rotate image
        rotated = rotate_image(image, angle)
        rotated_cv = np.array(rotated)
        rotated_cv = cv2.cvtColor(rotated_cv, cv2.COLOR_RGB2BGR)
        
        # Try OCR with confidence
        try:
            data = pytesseract.image_to_data(rotated_cv, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence of detected text
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                
                # Count words with high confidence (>60%)
                good_words = sum(1 for conf in confidences if conf > 60)
                
                # Count total words
                total_words = len([word for word in data['text'] if word.strip()])
                
                # Weighted score: average confidence * number of good words
                score = avg_confidence * good_words
                
                print(f"  Orientation {angle}°: score={score:.1f} (avg_conf={avg_confidence:.1f}%, good_words={good_words}, total_words={total_words})")
                
                if score > max_confidence:
                    max_confidence = score
                    best_orientation = angle
        except Exception as e:
            print(f"  OCR at {angle}° failed: {str(e)}")
    
    if max_confidence > 0:
        print(f"  Best orientation determined to be {best_orientation}° (score: {max_confidence:.1f})")
        return best_orientation
    
    # If all methods fail, return 0 (no rotation)
    print("  Could not determine orientation, defaulting to 0°")
    return 0

def verify_orientation(image, detected_angle):
    """
    Verify if the detected orientation makes sense by checking text line patterns.
    
    Args:
        image: PIL Image object
        detected_angle: The angle detected by the primary orientation detection
        
    Returns:
        int: Verified rotation angle (0, 90, 180, or 270)
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate horizontal and vertical projections
    h_proj = np.sum(thresh, axis=1)
    v_proj = np.sum(thresh, axis=0)
    
    # Calculate variance of projections
    h_var = np.var(h_proj)
    v_var = np.var(v_proj)
    
    # Calculate ratio of variances
    ratio = h_var / v_var if v_var > 0 else float('inf')
    
    print(f"  Projection analysis - H/V variance ratio: {ratio:.2f}")
    
    # For normal text (0 or 180 degrees), horizontal variance should be higher
    # For rotated text (90 or 270 degrees), vertical variance should be higher
    if ratio > 1.5:
        # Text is likely horizontal (0 or 180 degrees)
        if detected_angle in [90, 270]:
            print("  Orientation correction: Text appears to be horizontal, not vertical")
            return 0  # Default to 0 degrees
        else:
            return detected_angle
    elif ratio < 0.67:
        # Text is likely vertical (90 or 270 degrees)
        if detected_angle in [0, 180]:
            print("  Orientation correction: Text appears to be vertical, not horizontal")
            return 0  # Default to 0 degrees as we prefer not to rotate
        else:
            return detected_angle
    else:
        # Ratio is inconclusive, trust the detected angle
        return detected_angle

def rotate_image(image, angle):
    """
    Rotate the image by the given angle.
    
    Args:
        image: PIL Image object
        angle: Rotation angle in degrees
        
    Returns:
        PIL Image: Rotated image
    """
    if angle == 0:
        return image
    elif angle == 90:
        return image.transpose(Image.ROTATE_90)
    elif angle == 180:
        return image.transpose(Image.ROTATE_180)
    elif angle == 270:
        return image.transpose(Image.ROTATE_270)
    else:
        return image.rotate(angle, expand=True)

def enhance_image_quality(image):
    """
    Enhance the quality of low-quality images (like photos taken with a phone).
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Enhanced image
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Convert back to BGR for further processing
    enhanced_color = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB))

def preprocess_image(image, enhance_contrast=True):
    """
    Preprocess the image to improve OCR accuracy.
    
    Args:
        image: PIL Image object
        enhance_contrast: Whether to enhance contrast
        
    Returns:
        PIL Image: Preprocessed image
    """
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast if enabled
    if enhance_contrast:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Apply bilateral filter to preserve edges while removing noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Dilation followed by erosion to close small holes in the text
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Convert back to PIL Image
    return Image.fromarray(morph)

def post_process_text(text):
    """
    Post-process the extracted text to remove watermarks and improve quality.
    
    Args:
        text: Extracted text string
        
    Returns:
        str: Cleaned text
    """
    # Remove "Scanned with CamScanner" and similar watermarks
    text = re.sub(r'Scanned with CamScanner', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Scanned with\s+\w+', '', text, flags=re.IGNORECASE)
    
    # Remove any lines that only contain the watermark text
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if line.strip() and not re.search(r'^\s*CamScanner\s*$', line, re.IGNORECASE)]
    
    # Join the lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

def remove_noise_dots(image):
    """
    Remove small black dots/noise from the image while preserving text.
    
    Args:
        image: PIL Image object
        
    Returns:
        PIL Image: Image with noise dots removed
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    if len(img_cv.shape) == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale if it's not already
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv.copy()
    
    # Apply binary thresholding to separate text from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find all contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for text (larger contours)
    text_mask = np.zeros_like(binary)
    noise_mask = np.zeros_like(binary)
    
    # Parameters for noise detection
    min_text_area = 25  # Minimum area for text contours
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_text_area:
            # Small contours are likely noise dots
            cv2.drawContours(noise_mask, [contour], -1, 255, -1)
        else:
            # Larger contours are likely text
            cv2.drawContours(text_mask, [contour], -1, 255, -1)
    
    # Dilate the text mask slightly to ensure we don't remove parts of text
    kernel = np.ones((3, 3), np.uint8)
    text_mask_dilated = cv2.dilate(text_mask, kernel, iterations=1)
    
    # Remove noise that doesn't overlap with dilated text
    final_noise_mask = cv2.bitwise_and(noise_mask, cv2.bitwise_not(text_mask_dilated))
    
    # Create the cleaned binary image
    cleaned_binary = cv2.bitwise_and(binary, cv2.bitwise_not(final_noise_mask))
    
    # Invert back to get black text on white background
    cleaned = cv2.bitwise_not(cleaned_binary)
    
    # Convert back to PIL Image
    if len(img_cv.shape) == 3:
        # For color images, replace the cleaned areas in the original image
        mask_3channel = cv2.cvtColor(final_noise_mask, cv2.COLOR_GRAY2BGR)
        cleaned_color = img_cv.copy()
        cleaned_color[mask_3channel > 0] = [255, 255, 255]  # Set noise pixels to white
        return Image.fromarray(cv2.cvtColor(cleaned_color, cv2.COLOR_BGR2RGB))
    else:
        # For grayscale images
        return Image.fromarray(cleaned)

# NUOVA FUNZIONE: Separazione del testo principale dalle note
def separate_main_text_and_notes(image, extracted_text):
    """
    Separa il testo principale dalle note a piè di pagina.
    
    Args:
        image: PIL Image object
        extracted_text: Testo estratto dall'immagine
        
    Returns:
        tuple: (main_text, notes_text)
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    if len(img_cv.shape) == 3:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale if it's not already
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv.copy()
    
    # Get image dimensions
    height, width = gray.shape
    
    # Apply binary thresholding to separate text from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate horizontal projection to identify text lines
    h_proj = np.sum(binary, axis=1)
    
    # Normalize projection
    h_proj = h_proj / np.max(h_proj) if np.max(h_proj) > 0 else h_proj
    
    # Find potential note sections by analyzing text density and spacing
    # Notes typically have smaller font size, which means less dense text lines
    
    # Smooth the projection with a moving average filter
    window_size = 15
    h_proj_smooth = np.convolve(h_proj, np.ones(window_size)/window_size, mode='same')
    
    # Find significant gaps in text (potential separation between main text and notes)
    # A gap is defined as a region with low text density
    gap_threshold = 0.1 * np.max(h_proj_smooth)
    gaps = []
    
    in_gap = False
    gap_start = 0
    
    for i in range(len(h_proj_smooth)):
        if h_proj_smooth[i] < gap_threshold:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                in_gap = False
                gap_end = i
                # Only consider gaps that are significant (not just line spacing)
                if gap_end - gap_start > 10:  # Minimum gap size in pixels
                    gaps.append((gap_start, gap_end))
    
    # If we're still in a gap at the end of the image
    if in_gap:
        gaps.append((gap_start, len(h_proj_smooth)))
    
    # If no significant gaps were found, try to detect notes based on text density changes
    if not gaps and height > 200:
        # Analyze the bottom third of the page, where notes typically appear
        bottom_third_start = height * 2 // 3
        
        # Calculate average text density in the top 2/3 vs bottom 1/3
        top_density = np.mean(h_proj_smooth[:bottom_third_start])
        bottom_density = np.mean(h_proj_smooth[bottom_third_start:])
        
        # If bottom density is significantly lower, it might contain notes
        if bottom_density < 0.7 * top_density:
            # Look for the most significant drop in density
            density_drops = []
            
            for i in range(bottom_third_start, height - 20):
                before_avg = np.mean(h_proj_smooth[i-20:i])
                after_avg = np.mean(h_proj_smooth[i:i+20])
                
                if after_avg < 0.7 * before_avg:
                    density_drops.append((i, before_avg - after_avg))
            
            if density_drops:
                # Find the most significant drop
                max_drop = max(density_drops, key=lambda x: x[1])
                gaps.append((max_drop[0], max_drop[0] + 1))
    
    # Use OCR data to identify text regions and their properties
    try:
        # Get detailed OCR data including bounding boxes and font sizes
        ocr_data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
        
        # Extract bounding boxes, text, and confidence
        n_boxes = len(ocr_data['text'])
        boxes = []
        
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 0:  # Only consider text with confidence > 0
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                text = ocr_data['text'][i]
                
                if text.strip():  # Only consider non-empty text
                    boxes.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'text': text,
                        'area': w * h  # Calculate area as a proxy for font size
                    })
        
        # If we have boxes, analyze them to find potential notes
        if boxes:
            # Sort boxes by y-coordinate (top to bottom)
            boxes.sort(key=lambda box: box['y'])
            
            # Calculate median text area (proxy for font size)
            areas = [box['area'] for box in boxes]
            median_area = np.median(areas)
            
            # Identify potential note boxes (smaller font size)
            note_boxes = []
            main_boxes = []
            
            for box in boxes:
                # Notes typically have smaller font size (area < 0.8 * median)
                if box['area'] < 0.8 * median_area:
                    note_boxes.append(box)
                else:
                    main_boxes.append(box)
            
            # If we have identified note boxes
            if note_boxes:
                # Find the y-coordinate where notes start
                if len(note_boxes) > 3:  # Ensure we have enough note boxes for reliable detection
                    note_boxes.sort(key=lambda box: box['y'])
                    notes_start_y = note_boxes[0]['y']
                    
                    # Check if this matches any of our gap detections
                    if not gaps:
                        gaps.append((notes_start_y, height))
                    else:
                        # Update the closest gap
                        closest_gap_idx = min(range(len(gaps)), key=lambda i: abs(gaps[i][0] - notes_start_y))
                        gaps[closest_gap_idx] = (notes_start_y, gaps[closest_gap_idx][1])
    
    except Exception as e:
        print(f"  Warning: OCR analysis for note detection failed: {str(e)}")
    
    # If we found potential note sections, split the text
    main_text = extracted_text
    notes_text = ""
    
    if gaps:
        # Find the most significant gap (largest or closest to 2/3 of the page)
        if len(gaps) > 1:
            # Prefer gaps in the lower third of the page
            lower_third_gaps = [gap for gap in gaps if gap[0] > height * 2 // 3]
            
            if lower_third_gaps:
                # Choose the highest gap in the lower third
                main_note_separator = min(lower_third_gaps, key=lambda gap: gap[0])
            else:
                # Choose the gap closest to 2/3 of the page height
                main_note_separator = min(gaps, key=lambda gap: abs(gap[0] - height * 2 // 3))
        else:
            main_note_separator = gaps[0]
        
        # Use the gap to create a mask for main text and notes
        main_mask = np.zeros_like(binary)
        notes_mask = np.zeros_like(binary)
        
        # Main text is above the separator
        main_mask[:main_note_separator[0], :] = 255
        
        # Notes are below the separator
        notes_mask[main_note_separator[0]:, :] = 255
        
        # Apply masks to the original image
        main_region = cv2.bitwise_and(img_cv, img_cv, mask=main_mask)
        notes_region = cv2.bitwise_and(img_cv, img_cv, mask=notes_mask)
        
        # Extract text from each region
        try:
            # Extract text from main region
            main_text = pytesseract.image_to_string(main_region)
            
            # Extract text from notes region
            notes_text = pytesseract.image_to_string(notes_region)
            
            # Clean up the extracted text
            main_text = post_process_text(main_text)
            notes_text = post_process_text(notes_text)
        except Exception as e:
            print(f"  Warning: Region-based text extraction failed: {str(e)}")
            print(f"  Falling back to pattern-based text separation")
            
            # If OCR on regions fails, try to separate based on patterns in the text
            lines = extracted_text.split('\n')
            main_lines = []
            note_lines = []
            
            in_notes = False
            
            for line in lines:
                # Check for note patterns: numbers followed by period, superscript numbers, etc.
                if re.match(r'^\s*\d+[\.\)]', line) or re.match(r'^\s*\*', line) or re.match(r'^\s*[a-z][\.\)]', line):
                    in_notes = True
                
                if in_notes:
                    note_lines.append(line)
                else:
                    main_lines.append(line)
            
            main_text = '\n'.join(main_lines)
            notes_text = '\n'.join(note_lines)
    
    # If we still don't have a good separation, try pattern-based separation
    if not notes_text:
        # Look for patterns that indicate notes in the text
        lines = extracted_text.split('\n')
        main_lines = []
        note_lines = []
        
        in_notes = False
        
        for i, line in enumerate(lines):
            # Check for common note indicators
            is_note_start = False
            
            # Check for numbered notes (e.g., "33. Lo ricorda M. Sai...")
            if re.match(r'^\s*\d+[\.\)]', line):
                is_note_start = True
            
            # Check for asterisk notes (e.g., "* Cfr. La votazione...")
            if re.match(r'^\s*\*', line):
                is_note_start = True
            
            # Check for lettered notes (e.g., "a. Si veda il dattiloscritto...")
            if re.match(r'^\s*[a-z][\.\)]', line):
                is_note_start = True
            
            # Check for superscript indicators (e.g., "⁵ Per la biografia...")
            if re.search(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]', line):
                is_note_start = True
            
            # Check for page numbers (typically at the bottom of the page)
            if re.match(r'^\s*\d+\s*$', line):
                note_lines.append(line)
                continue
            
            # If we detect a note start pattern, switch to note mode
            if is_note_start:
                in_notes = True
            
            # Assign the line to the appropriate category
            if in_notes:
                note_lines.append(line)
            else:
                main_lines.append(line)
        
        # If we found notes, update the text variables
        if note_lines:
            main_text = '\n'.join(main_lines)
            notes_text = '\n'.join(note_lines)
    
    return main_text, notes_text

def process_image(image, page_num, output_dir=None, auto_orient=True, preprocess=True, 
                 remove_fingers=True, remove_watermarks=True, correct_lighting=True,
                 correct_color=True, enhance_quality=True, save_processed_images=False,
                 enhance_contrast=True, conservative_finger_removal=True,
                 remove_noise=True, separate_notes=True):  # Nuovo parametro
    """
    Process a single image through the entire pipeline.
    
    Args:
        image: PIL Image object
        page_num: Page number for logging
        output_dir: Directory to save processed images
        auto_orient: Whether to automatically detect and correct page orientation
        preprocess: Whether to preprocess images to improve OCR accuracy
        remove_fingers: Whether to attempt to remove fingers from scanned images
        remove_watermarks: Whether to attempt to remove watermarks from images
        correct_lighting: Whether to correct uneven lighting and shadows
        correct_color: Whether to correct color casts like bluish tints
        enhance_quality: Whether to enhance quality of low-quality images
        save_processed_images: Whether to save the processed images for debugging
        enhance_contrast: Whether to enhance contrast during preprocessing
        conservative_finger_removal: Whether to use a more conservative approach for finger removal
        remove_noise: Whether to remove small noise dots from the image
        separate_notes: Whether to separate main text from notes
        
    Returns:
        tuple: (processed_image, main_text, notes_text)
    """
    print(f"Processing page {page_num}...")
    
    # Save original image if enabled
    if save_processed_images and output_dir:
        original_image_path = os.path.join(output_dir, f"page_{page_num:03d}_original.png")
        image.save(original_image_path)
        print(f"  Saved original image to: {original_image_path}")
    
    # Create a copy of the original image for processing
    processed_image = image.copy()
    
    # Enhance image quality if enabled (for photos taken with phones)
    if enhance_quality:
        try:
            print(f"  Enhancing image quality for page {page_num}...")
            processed_image = enhance_image_quality(processed_image)
            print(f"  Image quality enhancement completed")
            
            if save_processed_images and output_dir:
                enhanced_image_path = os.path.join(output_dir, f"page_{page_num:03d}_enhanced.png")
                processed_image.save(enhanced_image_path)
                print(f"  Saved enhanced image to: {enhanced_image_path}")
        except Exception as e:
            print(f"  Warning: Image quality enhancement failed for page {page_num}: {str(e)}")
            print(f"  Proceeding with original image")
    
    # Correct color cast if enabled
    if correct_color:
        try:
            print(f"  Checking for color cast in page {page_num}...")
            processed_image = correct_color_cast(processed_image)
            
            if save_processed_images and output_dir:
                color_corrected_path = os.path.join(output_dir, f"page_{page_num:03d}_color_corrected.png")
                processed_image.save(color_corrected_path)
                print(f"  Saved color-corrected image to: {color_corrected_path}")
        except Exception as e:
            print(f"  Warning: Color correction failed for page {page_num}: {str(e)}")
            print(f"  Proceeding with original image")
    
    # Correct uneven lighting if enabled
    if correct_lighting:
        try:
            print(f"  Correcting uneven lighting for page {page_num}...")
            processed_image = correct_uneven_lighting(processed_image)
            print(f"  Lighting correction completed")
            
            if save_processed_images and output_dir:
                lighting_corrected_path = os.path.join(output_dir, f"page_{page_num:03d}_lighting_corrected.png")
                processed_image.save(lighting_corrected_path)
                print(f"  Saved lighting-corrected image to: {lighting_corrected_path}")
        except Exception as e:
            print(f"  Warning: Lighting correction failed for page {page_num}: {str(e)}")
            print(f"  Proceeding with original image")
    
    # Remove watermarks if enabled
    if remove_watermarks:
        try:
            print(f"  Attempting to remove watermarks from page {page_num}...")
            processed_image = remove_watermark(processed_image)
            print(f"  Watermark removal completed")
            
            if save_processed_images and output_dir:
                watermark_removed_path = os.path.join(output_dir, f"page_{page_num:03d}_watermark_removed.png")
                processed_image.save(watermark_removed_path)
                print(f"  Saved watermark-removed image to: {watermark_removed_path}")
        except Exception as e:
            print(f"  Warning: Watermark removal failed for page {page_num}: {str(e)}")
            print(f"  Proceeding with original image")
    
    # Auto-orient the image if enabled
    if auto_orient:
        try:
            print(f"  Detecting orientation for page {page_num}...")
            angle = detect_orientation_robust(processed_image)
            
            # Verify the orientation to avoid incorrect 90/270 degree rotations
            verified_angle = verify_orientation(processed_image, angle)
            
            if verified_angle != angle:
                print(f"  Orientation corrected from {angle}° to {verified_angle}°")
                angle = verified_angle
            
            if angle != 0:
                print(f"  Rotating page {page_num} by {angle} degrees")
                processed_image = rotate_image(processed_image, angle)
                
                if save_processed_images and output_dir:
                    rotated_image_path = os.path.join(output_dir, f"page_{page_num:03d}_rotated.png")
                    processed_image.save(rotated_image_path)
                    print(f"  Saved rotated image to: {rotated_image_path}")
            else:
                print(f"  Page {page_num} has correct orientation")
        except Exception as e:
            print(f"  Warning: Could not detect orientation for page {page_num}: {str(e)}")
            print(f"  Proceeding with original orientation")
    
    # Remove fingers if enabled
    if remove_fingers:
        try:
            print(f"  Attempting to remove fingers from page {page_num}...")
            processed_image = detect_fingers(processed_image, conservative=conservative_finger_removal)
            print(f"  Finger removal completed")
            
            if save_processed_images and output_dir:
                fingers_removed_path = os.path.join(output_dir, f"page_{page_num:03d}_fingers_removed.png")
                processed_image.save(fingers_removed_path)
                print(f"  Saved fingers-removed image to: {fingers_removed_path}")
        except Exception as e:
            print(f"  Warning: Finger removal failed for page {page_num}: {str(e)}")
            print(f"  Proceeding with original image")
    
    # Remove noise dots if enabled
    if remove_noise:
        try:
            print(f"  Removing noise dots from page {page_num}...")
            processed_image = remove_noise_dots(processed_image)
            print(f"  Noise removal completed")
            
            if save_processed_images and output_dir:
                noise_removed_path = os.path.join(output_dir, f"page_{page_num:03d}_noise_removed.png")
                processed_image.save(noise_removed_path)
                print(f"  Saved noise-removed image to: {noise_removed_path}")
        except Exception as e:
            print(f"  Warning: Noise removal failed for page {page_num}: {str(e)}")
            print(f"  Proceeding with original image")
    
    # Preprocess the image if enabled
    if preprocess:
        print(f"  Preprocessing page {page_num}...")
        try:
            processed_image = preprocess_image(processed_image, enhance_contrast=enhance_contrast)
            print(f"  Preprocessing completed")
            
            if save_processed_images and output_dir:
                preprocessed_image_path = os.path.join(output_dir, f"page_{page_num:03d}_preprocessed.png")
                processed_image.save(preprocessed_image_path)
                print(f"  Saved preprocessed image to: {preprocessed_image_path}")
        except Exception as e:
            print(f"  Warning: Preprocessing failed for page {page_num}: {str(e)}")
            print(f"  Proceeding with original image")
    
    # Save final processed image if enabled
    if save_processed_images and output_dir:
        final_image_path = os.path.join(output_dir, f"page_{page_num:03d}_final.png")
        processed_image.save(final_image_path)
        print(f"  Saved final processed image to: {final_image_path}")
    
    # Use pytesseract to extract text from the image
    extracted_text = ""
    main_text = ""
    notes_text = ""
    
    try:
        # Try multiple OCR configurations and use the best result
        configs = [
            '--oem 1 --psm 3',  # Default: Automatic page segmentation with OSD
            '--oem 1 --psm 6',  # Assume a single uniform block of text
            '--oem 1 --psm 4',  # Assume a single column of text
            # Add Italian language support
            '--oem 1 --psm 3 -l ita',  # Italian language with automatic page segmentation
        ]
        
        best_text = ""
        best_word_count = 0
        
        for config in configs:
            page_text = pytesseract.image_to_string(processed_image, config=config)
            word_count = len([w for w in page_text.split() if len(w) > 1])
            
            if word_count > best_word_count:
                best_text = page_text
                best_word_count = word_count
        
        # Post-process the text to remove watermarks and improve quality
        best_text = post_process_text(best_text)
        
        extracted_text = best_text
        print(f"  Extracted {best_word_count} words from page {page_num}")
        
        # Separate main text from notes if enabled
        if separate_notes:
            try:
                print(f"  Separating main text from notes for page {page_num}...")
                main_text, notes_text = separate_main_text_and_notes(processed_image, extracted_text)
                
                main_word_count = len([w for w in main_text.split() if len(w) > 1])
                notes_word_count = len([w for w in notes_text.split() if len(w) > 1])
                
                print(f"  Separated text: {main_word_count} words in main text, {notes_word_count} words in notes")
                
                # If no notes were detected, use the full text as main text
                if not notes_text.strip():
                    main_text = extracted_text
                    print(f"  No notes detected, using full text as main text")
            except Exception as e:
                print(f"  Warning: Text separation failed for page {page_num}: {str(e)}")
                print(f"  Using full text without separation")
                main_text = extracted_text
        else:
            main_text = extracted_text
    except Exception as e:
        print(f"  Error: OCR failed for page {page_num}: {str(e)}")
        extracted_text = f"[OCR FAILED: {str(e)}]"
        main_text = extracted_text
    
    return processed_image, main_text, notes_text

def extract_text_from_scanned_pdf(pdf_path, output_txt_path=None, output_main_txt_path=None, output_notes_txt_path=None,
                                 auto_orient=True, preprocess=True, remove_fingers=True, remove_watermarks=True, 
                                 correct_lighting=True, correct_color=True, enhance_quality=True, 
                                 save_processed_images=False, output_dir=None, enhance_contrast=True, 
                                 conservative_finger_removal=True, skip_incomplete_pages=True, 
                                 split_double_pages=True, remove_noise=True, separate_notes=True):  # Nuovi parametri
    """
    Extract text from a scanned PDF file using OCR.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_txt_path (str, optional): Path to save the extracted text
        output_main_txt_path (str, optional): Path to save the main text
        output_notes_txt_path (str, optional): Path to save the notes text
        auto_orient (bool): Whether to automatically detect and correct page orientation
        preprocess (bool): Whether to preprocess images to improve OCR accuracy
        remove_fingers (bool): Whether to attempt to remove fingers from scanned images
        remove_watermarks (bool): Whether to attempt to remove watermarks from images
        correct_lighting (bool): Whether to correct uneven lighting and shadows
        correct_color (bool): Whether to correct color casts like bluish tints
        enhance_quality (bool): Whether to enhance quality of low-quality images
        save_processed_images (bool): Whether to save the processed images for debugging
        output_dir (str, optional): Directory to save processed images
        enhance_contrast (bool): Whether to enhance contrast during preprocessing
        conservative_finger_removal (bool): Whether to use a more conservative approach for finger removal
        skip_incomplete_pages (bool): Whether to skip pages that are detected as incomplete/cut off
        split_double_pages (bool): Whether to split images containing two pages into separate pages
        remove_noise (bool): Whether to remove small noise dots from the image
        separate_notes (bool): Whether to separate main text from notes
        
    Returns:
        tuple: (extracted_text, main_text, notes_text)
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Create output directory for processed images if needed
    if save_processed_images:
        if output_dir is None:
            output_dir = os.path.splitext(pdf_path)[0] + "_processed_images"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processed images will be saved to: {output_dir}")
    
    # Convert PDF to images
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path)
    print(f"PDF converted to {len(images)} images")
    
    # Extract text from each image using OCR
    print("Extracting text using OCR (this may take some time)...")
    extracted_text = ""
    main_text_all = ""
    notes_text_all = ""
    
    page_counter = 1
    
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...")
        
        # Check if the page is complete or cut off
        is_complete, completeness_score = detect_page_completeness(image)
        
        if not is_complete and skip_incomplete_pages:
            print(f"  Image {i+1} appears to be incomplete (score: {completeness_score:.2f}), skipping")
            extracted_text += f"\n\n--- PAGE {page_counter} (SKIPPED - INCOMPLETE) ---\n\n"
            main_text_all += f"\n\n--- PAGE {page_counter} (SKIPPED - INCOMPLETE) ---\n\n"
            notes_text_all += f"\n\n--- PAGE {page_counter} (SKIPPED - INCOMPLETE) ---\n\n"
            page_counter += 1
            continue
        
        # Check if the image contains two pages
        if split_double_pages:
            pages, is_double_page = detect_double_page(image)
            
            if is_double_page:
                print(f"  Image {i+1} contains two pages, splitting")
                
                for j, page in enumerate(pages):
                    page_label = f"{page_counter} (Left)" if j == 0 else f"{page_counter} (Right)"
                    
                    # Process the individual page
                    _, page_main_text, page_notes_text = process_image(
                        page, f"{page_counter} ({j+1}/2)", output_dir,
                        auto_orient, preprocess, remove_fingers, remove_watermarks,
                        correct_lighting, correct_color, enhance_quality,
                        save_processed_images, enhance_contrast, conservative_finger_removal,
                        remove_noise, separate_notes  # Passa i nuovi parametri
                    )
                    
                    # Combine the main text and notes for the full extracted text
                    page_full_text = page_main_text
                    if page_notes_text:
                        page_full_text += "\n\n--- NOTES ---\n\n" + page_notes_text
                    
                    extracted_text += f"\n\n--- PAGE {page_label} ---\n\n"
                    extracted_text += page_full_text
                    
                    main_text_all += f"\n\n--- PAGE {page_label} ---\n\n"
                    main_text_all += page_main_text
                    
                    notes_text_all += f"\n\n--- PAGE {page_label} NOTES ---\n\n"
                    notes_text_all += page_notes_text
                    
                    page_counter += 1
            else:
                # Process the single page
                _, page_main_text, page_notes_text = process_image(
                    image, page_counter, output_dir,
                    auto_orient, preprocess, remove_fingers, remove_watermarks,
                    correct_lighting, correct_color, enhance_quality,
                    save_processed_images, enhance_contrast, conservative_finger_removal,
                    remove_noise, separate_notes  # Passa i nuovi parametri
                )
                
                # Combine the main text and notes for the full extracted text
                page_full_text = page_main_text
                if page_notes_text:
                    page_full_text += "\n\n--- NOTES ---\n\n" + page_notes_text
                
                extracted_text += f"\n\n--- PAGE {page_counter} ---\n\n"
                extracted_text += page_full_text
                
                main_text_all += f"\n\n--- PAGE {page_counter} ---\n\n"
                main_text_all += page_main_text
                
                notes_text_all += f"\n\n--- PAGE {page_counter} NOTES ---\n\n"
                notes_text_all += page_notes_text
                
                page_counter += 1
        else:
            # Process the single page without checking for double pages
            _, page_main_text, page_notes_text = process_image(
                image, page_counter, output_dir,
                auto_orient, preprocess, remove_fingers, remove_watermarks,
                correct_lighting, correct_color, enhance_quality,
                save_processed_images, enhance_contrast, conservative_finger_removal,
                remove_noise, separate_notes  # Passa i nuovi parametri
            )
            
            # Combine the main text and notes for the full extracted text
            page_full_text = page_main_text
            if page_notes_text:
                page_full_text += "\n\n--- NOTES ---\n\n" + page_notes_text
            
            extracted_text += f"\n\n--- PAGE {page_counter} ---\n\n"
            extracted_text += page_full_text
            
            main_text_all += f"\n\n--- PAGE {page_counter} ---\n\n"
            main_text_all += page_main_text
            
            notes_text_all += f"\n\n--- PAGE {page_counter} NOTES ---\n\n"
            notes_text_all += page_notes_text
            
            page_counter += 1
    
    # Save the extracted text to files if output paths are provided
    if output_txt_path:
        print(f"Saving extracted text to: {output_txt_path}")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
    
    if output_main_txt_path:
        print(f"Saving main text to: {output_main_txt_path}")
        with open(output_main_txt_path, 'w', encoding='utf-8') as f:
            f.write(main_text_all)
    
    if output_notes_txt_path:
        print(f"Saving notes text to: {output_notes_txt_path}")
        with open(output_notes_txt_path, 'w', encoding='utf-8') as f:
            f.write(notes_text_all)
    
    print("Text extraction completed!")
    return extracted_text, main_text_all, notes_text_all

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text from scanned PDF using OCR')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Path to save the extracted text (optional)')
    parser.add_argument('--output-main', help='Path to save the main text (optional)')
    parser.add_argument('--output-notes', help='Path to save the notes text (optional)')
    parser.add_argument('--no-auto-orient', action='store_true', 
                        help='Disable automatic orientation detection and correction')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Disable image preprocessing')
    parser.add_argument('--no-finger-removal', action='store_true',
                        help='Disable finger removal from scanned images')
    parser.add_argument('--no-watermark-removal', action='store_true',
                        help='Disable watermark removal from images')
    parser.add_argument('--no-lighting-correction', action='store_true',
                        help='Disable correction of uneven lighting and shadows')
    parser.add_argument('--no-color-correction', action='store_true',
                        help='Disable correction of color casts')
    parser.add_argument('--no-quality-enhancement', action='store_true',
                        help='Disable enhancement of low-quality images')
    parser.add_argument('--no-enhance-contrast', action='store_true',
                        help='Disable contrast enhancement during preprocessing')
    parser.add_argument('--aggressive-finger-removal', action='store_true',
                        help='Use more aggressive finger removal (less conservative)')
    parser.add_argument('--no-skip-incomplete', action='store_true',
                        help='Process all pages, even if they appear to be incomplete')
    parser.add_argument('--no-split-double', action='store_true',
                        help='Do not split images containing two pages')
    parser.add_argument('--save-images', action='store_true',
                        help='Save processed images for debugging')
    parser.add_argument('--image-dir', help='Directory to save processed images (optional)')
    parser.add_argument('--no-noise-removal', action='store_true',
                        help='Disable removal of small noise dots from images')
    parser.add_argument('--no-separate-notes', action='store_true',
                        help='Disable separation of main text from notes')
    parser.add_argument('--no-chatgpt', action='store_true', 
                        help='Disabilita il miglioramento con ChatGPT')
    parser.add_argument('--chatgpt-format', choices=['clean_text', 'academic', 'formal', 'markdown'],
                        default='clean_text', help='Formato di output per ChatGPT')

    args = parser.parse_args()
    
    # If output paths are not provided, create them based on the PDF filename
    if not args.output:
        output_path = os.path.splitext(args.pdf_path)[0] + '_text.txt'
    else:
        output_path = args.output
    
    if not args.output_main:
        output_main_path = os.path.splitext(args.pdf_path)[0] + '_main_text.txt'
    else:
        output_main_path = args.output_main
    
    if not args.output_notes:
        output_notes_path = os.path.splitext(args.pdf_path)[0] + '_notes.txt'
    else:
        output_notes_path = args.output_notes
    
    # Extract text from the PDF
    extract_text_from_scanned_pdf_with_chatgpt(
        args.pdf_path, 
        output_path,
        output_main_path,
        output_notes_path,
        not args.no_auto_orient,
        not args.no_preprocess,
        not args.no_finger_removal,
        not args.no_watermark_removal,
        not args.no_lighting_correction,
        not args.no_color_correction,
        not args.no_quality_enhancement,
        args.save_images,
        args.image_dir,
        not args.no_enhance_contrast,
        not args.aggressive_finger_removal,
        not args.no_skip_incomplete,
        not args.no_split_double,
        not args.no_noise_removal,
        not args.no_separate_notes,
        not args.no_chatgpt,
        args.chatgpt_format

    )
    print(f"Extracted text saved to: {output_path}")
    print(f"Main text saved to: {output_main_path}")
    print(f"Notes text saved to: {output_notes_path}")