import pandas as pd
import matplotlib.pyplot as plt

# Datos
palabras = [
    'Google', 'Facebook', 'Twitter', 'Apple', 'World', 'Video', 'App', 'Watch', 'Iphone', 'Game', 'Know', 'Time', 'Social'
]
frecuencias = [
    1505, 1249, 1218, 1166, 1077, 972, 868, 725, 687, 684, 640, 633, 628
]

# Crear DataFrame
df = pd.DataFrame({'Palabra': palabras, 'Frecuencia': frecuencias})

# Mostrar tabla sencilla
print(df)


# Gráfico de barras (torrecitas)
plt.figure(figsize=(10, 6))
plt.bar(df['Palabra'], df['Frecuencia'], color='skyblue')
plt.xlabel('Palabra')
plt.ylabel('Frecuencia')
plt.title('Frecuencia de palabras en títulos de noticias')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
