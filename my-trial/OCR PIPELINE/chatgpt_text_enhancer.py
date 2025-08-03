"""
Modulo per migliorare il testo estratto dall'OCR usando ChatGPT
"""

from openai import OpenAI
import os
import re
import json
from typing import Tuple, Optional, Dict, Any
import time

class ChatGPTTextEnhancer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Inizializza l'enhancer con la chiave API di OpenAI
        
        Args:
            api_key: Chiave API OpenAI (se None, la legge dalle variabili d'ambiente)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key OpenAI non trovata. Impostala come variabile d'ambiente OPENAI_API_KEY o passala al costruttore.")
        
        # Configura il client OpenAI (NUOVA API)
        self.client = OpenAI(api_key=self.api_key)
        
        # Parametri di configurazione
        self.model = "gpt-4"  # Usa GPT-4 per migliori risultati
        self.max_tokens = 4000
        self.temperature = 0.3  # Bassa temperatura per risultati più consistenti
        
    def enhance_text(self, text: str, text_type: str = "general", 
                    target_format: str = "clean_text") -> Tuple[str, Dict[str, Any]]:
        """
        Migliora il testo estratto dall'OCR usando ChatGPT
        
        Args:
            text: Testo da migliorare
            text_type: Tipo di testo ("main", "notes", "general")
            target_format: Formato desiderato ("clean_text", "academic", "formal", "markdown")
            
        Returns:
            Tuple[str, Dict]: (testo_migliorato, metadati_analisi)
        """
        if not text or not text.strip():
            return text, {"status": "empty", "changes": 0}
        
        print(f"  Migliorando testo con ChatGPT ({len(text)} caratteri)...")
        
        try:
            # Crea il prompt basato sul tipo di testo e formato richiesto
            prompt = self._create_enhancement_prompt(text, text_type, target_format)
            
            # Chiama l'API OpenAI (NUOVA SINTASSI)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(text_type, target_format)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Estrai la risposta (NUOVA STRUTTURA)
            enhanced_text = response.choices[0].message.content.strip()
            
            # Analizza i cambiamenti
            analysis = self._analyze_changes(text, enhanced_text)
            
            # Aggiungi metadati della risposta
            analysis.update({
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens,
                "status": "success"
            })
            
            print(f"  ✅ Testo migliorato: {analysis['changes']} modifiche, {analysis['tokens_used']} token utilizzati")
            
            return enhanced_text, analysis
            
        except Exception as e:
            print(f"  ❌ Errore nell'enhancement: {str(e)}")
            return text, {"status": "error", "error": str(e), "changes": 0}
    
    def _get_system_prompt(self, text_type: str, target_format: str) -> str:
        """Crea il prompt di sistema basato sul tipo di testo e formato"""
        
        base_prompt = """Sei un esperto editor di testi specializzato nel miglioramento di testi estratti tramite OCR. 
Il tuo compito è correggere errori, migliorare la formattazione e rendere il testo più leggibile mantenendo il significato originale."""
        
        type_specific = {
            "main": "Stai lavorando su un testo principale. Concentrati sulla coerenza narrativa e la fluidità di lettura.",
            "notes": "Stai lavorando su note a piè di pagina o riferimenti. Mantieni la struttura delle citazioni e numerazioni.",
            "general": "Stai lavorando su un testo generico. Applica correzioni standard."
        }
        
        format_specific = {
            "clean_text": "Produci un testo pulito e ben formattato, senza markup speciali.",
            "academic": "Mantieni lo stile accademico con citazioni e riferimenti appropriati.",
            "formal": "Usa un registro formale e professionale.",
            "markdown": "Formatta il testo usando la sintassi Markdown appropriata."
        }
        
        return f"{base_prompt}\n\n{type_specific.get(text_type, type_specific['general'])}\n\n{format_specific.get(target_format, format_specific['clean_text'])}"
    
    def _create_enhancement_prompt(self, text: str, text_type: str, target_format: str) -> str:
        """Crea il prompt per il miglioramento del testo"""
        
        prompt = f"""Migliora il seguente testo estratto tramite OCR seguendo queste linee guida:

1. **Correzioni OCR**: Correggi errori tipici dell'OCR come caratteri mal riconosciuti, parole spezzate, spaziature errate
2. **Caratteri speciali**: Rimuovi o correggi caratteri speciali non conformi (es: □, ■, ◊, caratteri di controllo)
3. **Formattazione**: Migliora la struttura del testo, paragrafi, spaziature
4. **Coerenza semantica**: Assicurati che il testo abbia senso logico e correggi incongruenze
5. **Punteggiatura**: Correggi punteggiatura mancante o errata
6. **Maiuscole/minuscole**: Correggi l'uso inappropriato di maiuscole e minuscole

IMPORTANTE: 
- Mantieni il significato originale del testo
- Non aggiungere informazioni non presenti nel testo originale
- Se una parola è ambigua, scegli l'interpretazione più probabile nel contesto
- Preserva la struttura originale (paragrafi, elenchi, etc.)

Testo da migliorare:
---
{text}
---

Fornisci SOLO il testo migliorato, senza spiegazioni o commenti aggiuntivi."""
        
        return prompt
    
    def _analyze_changes(self, original: str, enhanced: str) -> Dict[str, Any]:
        """Analizza i cambiamenti apportati al testo"""
        
        # Conta le differenze di base
        original_words = len(original.split())
        enhanced_words = len(enhanced.split())
        original_chars = len(original)
        enhanced_chars = len(enhanced)
        
        # Stima il numero di cambiamenti (semplificata)
        changes = abs(original_chars - enhanced_chars) + abs(original_words - enhanced_words)
        
        # Rileva miglioramenti specifici
        improvements = {
            "removed_special_chars": len(re.findall(r'[□■◊▪▫◦‣⁃]', original)) - len(re.findall(r'[□■◊▪▫◦‣⁃]', enhanced)),
            "fixed_spacing": original.count('  ') - enhanced.count('  '),
            "added_punctuation": enhanced.count('.') + enhanced.count(',') + enhanced.count(';') - 
                               (original.count('.') + original.count(',') + original.count(';'))
        }
        
        return {
            "changes": changes,
            "original_words": original_words,
            "enhanced_words": enhanced_words,
            "original_chars": original_chars,
            "enhanced_chars": enhanced_chars,
            "improvements": improvements
        }
    
    def enhance_text_in_chunks(self, text: str, chunk_size: int = 3000, 
                              text_type: str = "general", target_format: str = "clean_text") -> Tuple[str, Dict[str, Any]]:
        """
        Migliora testi lunghi dividendoli in chunk per evitare limiti di token
        
        Args:
            text: Testo da migliorare
            chunk_size: Dimensione massima di ogni chunk in caratteri
            text_type: Tipo di testo
            target_format: Formato desiderato
            
        Returns:
            Tuple[str, Dict]: (testo_migliorato, metadati_analisi)
        """
        if len(text) <= chunk_size:
            return self.enhance_text(text, text_type, target_format)
        
        print(f"  Testo lungo ({len(text)} caratteri), dividendo in chunk...")
        
        # Dividi il testo in chunk intelligenti (per paragrafi quando possibile)
        chunks = self._split_text_intelligently(text, chunk_size)
        
        enhanced_chunks = []
        total_analysis = {
            "chunks_processed": len(chunks),
            "total_changes": 0,
            "total_tokens_used": 0,
            "status": "success"
        }
        
        for i, chunk in enumerate(chunks):
            print(f"  Processando chunk {i+1}/{len(chunks)}...")
            
            enhanced_chunk, analysis = self.enhance_text(chunk, text_type, target_format)
            enhanced_chunks.append(enhanced_chunk)
            
            # Accumula le statistiche
            total_analysis["total_changes"] += analysis.get("changes", 0)
            total_analysis["total_tokens_used"] += analysis.get("tokens_used", 0)
            
            # Pausa breve per evitare rate limiting
            if i < len(chunks) - 1:
                time.sleep(1)
        
        # Unisci i chunk migliorati
        enhanced_text = "\n\n".join(enhanced_chunks)
        
        return enhanced_text, total_analysis
    
    def _split_text_intelligently(self, text: str, max_size: int) -> list:
        """Divide il testo in chunk intelligenti rispettando paragrafi e frasi"""
        
        chunks = []
        current_chunk = ""
        
        # Prima prova a dividere per paragrafi
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # Se il paragrafo da solo è troppo grande, dividilo per frasi
            if len(paragraph) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # Anche la singola frase è troppo lunga, dividila forzatamente
                            chunks.append(sentence[:max_size])
                            current_chunk = sentence[max_size:]
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
            else:
                # Il paragrafo è di dimensione accettabile
                if len(current_chunk) + len(paragraph) > max_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph
                    else:
                        chunks.append(paragraph)
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Aggiungi l'ultimo chunk se non vuoto
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

def test_enhancer():
    """Funzione di test per l'enhancer"""
    
    # Testo di esempio con errori tipici dell'OCR
    test_text = """
    Questo è un testo di prova con errori tipici dell'OCR.
    Ci sono caratteri strani come □ e ■ che dovrebbero essere rimossi.
    Alcune parole sono spa zzate in modo errato.
    mancano maiuscole all'inizio delle frasi e la punteggiatura,è sbagliata
    
    Questo paragrafo ha    spaziature    multiple    errate.
    
    ◊ Questo punto elenco ha un carattere speciale sbagliato
    ▪ Anche questo
    """
    
    try:
        enhancer = ChatGPTTextEnhancer()
        enhanced, analysis = enhancer.enhance_text(test_text, "general", "clean_text")
        
        print("=== TESTO ORIGINALE ===")
        print(test_text)
        print("\n=== TESTO MIGLIORATO ===")
        print(enhanced)
        print("\n=== ANALISI ===")
        print(json.dumps(analysis, indent=2))
        
    except Exception as e:
        print(f"Errore nel test: {e}")

if __name__ == "__main__":
    test_enhancer()
