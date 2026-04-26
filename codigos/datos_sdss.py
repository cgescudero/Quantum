from astroquery.sdss import SDSS
import pandas as pd
import io

# 1. Ajustamos el timeout por si el servidor de SDSS está lento
SDSS.TIMEOUT = 120 

# 2. Corregimos [class] y las columnas de [OII]
query = """
SELECT TOP 1000
    s.specobjid, s.ra, s.dec, s.z,
    l.h_alpha_flux, l.h_alpha_flux_err,
    l.h_beta_flux, l.h_beta_flux_err,
    l.oiii_5007_flux, l.oiii_5007_flux_err,
    l.nii_6584_flux, l.nii_6584_flux_err,
    l.oii_3726_flux, l.oii_3729_flux,
    l.sii_6717_flux, l.sii_6731_flux,
    e.bptclass, e.lgm_tot_p50 as log_stellar_mass, 
    e.oh_p50 as metallicity
FROM SpecObj AS s
JOIN galSpecLine AS l ON s.specobjid = l.specobjid
JOIN galSpecExtra AS e ON s.specobjid = e.specobjid
WHERE 
    s.[class] = 'GALAXY' 
    AND s.z BETWEEN 0.02 AND 0.1
    AND l.h_alpha_flux > 10
    AND l.h_beta_flux > 5
    AND e.bptclass = 1
    AND e.oh_p50 > 0
"""

print("Enviando consulta al SDSS (esto puede tomar unos segundos)...")

try:
    # Usamos query_sql_async para capturar la respuesta "cruda" del servidor
    # Especificamos explícitamente el Data Release 17 (donde MPA-JHU es muy estable)
    response = SDSS.query_sql_async(query, data_release=17)
    
    if response.status_code == 200:
        texto_respuesta = response.text.strip()
        
        # Comprobamos si el SDSS nos devolvió un HTML (un error)
        if texto_respuesta.startswith("<"):
            print("\n❌ EL SERVIDOR SDSS RECHAZÓ LA CONSULTA. Mensaje de error real:")
            # Buscamos dónde empieza el error real en el HTML (suele decir "SQL statement")
            print(texto_respuesta[:1000])  # Imprimimos los primeros 1000 caracteres
        else:
            # Si no es HTML, son los datos correctos separados por comas
            # Pandas lee directamente el texto en memoria
            df = pd.read_csv(io.StringIO(texto_respuesta), comment='#')
            
            # Guardamos el archivo
            df.to_csv('sdss_galaxies_spectroscopy.csv', index=False)
            print(f"\n✅ ¡Éxito! Se descargaron {len(df)} galaxias y se guardaron en 'sdss_galaxies_spectroscopy.csv'.")
    else:
        print(f"Error de conexión. El servidor devolvió el código: {response.status_code}")

except Exception as e:
    print(f"Ocurrió un error en Python: {e}")