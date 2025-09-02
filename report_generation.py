import ollama

def estrai_etichette_e_trascrizioni(percorso_file):
    """Legge il file di trascrizione ed estrae solo le parti del Dottore e Paziente."""
    conversazione = []
    with open(percorso_file, "r", encoding="utf-8") as file:
        for linea in file:
            linea = linea.strip()
            if "Dottore:" in linea or "Paziente:" in linea:
                # Divide la linea nelle sue componenti principali
                parti = linea.split("|")
                if len(parti) >= 2:
                    parte_utile = parti[1].strip()
                    dettagli = parte_utile.split(":", 2)
                    if len(dettagli) >= 3:
                        parlante = dettagli[1].strip()
                        testo = dettagli[2].strip()
                        conversazione.append(f"{parlante}: {testo}")
    return "\n".join(conversazione)

def genera_report_medico(conversazione):
    """Genera un'analisi strutturata della conversazione medica."""
    prompt = f"""
Analizza questa conversazione clinica ed estrai TUTTI i dati medici rilevanti forniti dal Dottore.
Organizza il report in queste categorie:

1. Valori clinici anomali:
   - Parametro | Valore | Note

2. Risultati esami strumentali:
   - Tipo esame | Risultato | Notehow to be a businessman


Usa solo informazioni esplicite dal Dottore. Non includere informazioni del paziente.
Mantieni i valori numerici o le valutazioni qualitative originali (es. "leggermente elevato").
Formatta tutto in markdown con elenchi puntati.

Conversazione:
{conversazione}

Report Medico:
"""
    try:
        risposta = ollama.chat(
            model="gemma3:12b",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        if 'message' in risposta and 'content' in risposta['message']:
            report = risposta['message']['content']
            with open("report_medico.md", "w", encoding="utf-8") as file:
                file.write(report)
            return report
        return "Errore nella generazione del report"
    
    except Exception as e:
        return f"Errore: {str(e)}"

def main():
    conversazione = estrai_etichette_e_trascrizioni("global_transcript.txt")
    if not conversazione:
        print("Nessuna conversazione trovata")
        return
    
    print("Analisi in corso...")
    report = genera_report_medico(conversazione)
    
    if report:
        print("\nReport generato con successo:")
        print(report)
    else:
        print("Errore nella generazione del report")

if __name__ == "__main__":
    main()
