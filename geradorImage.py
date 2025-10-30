# Programa para gerar imagens de teste para Template Matching
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, transform, filters, exposure
from skimage.io import imsave
import cv2
from google.colab import files
import os

print("=== GERADOR DE IMAGENS PARA TESTE DE TEMPLATE MATCHING ===")

def criar_imagem_com_objetos_geometricos():
    """Cria uma imagem com múltiplos objetos geométricos"""
    # Criar imagem base
    imagem = np.ones((400, 600)) * 128  # Fundo cinza
    
    # Adicionar diferentes formas geométricas
    # Círculos
    rr, cc = draw.disk((100, 100), 30)
    imagem[rr, cc] = 255
    
    rr, cc = draw.disk((300, 150), 25)
    imagem[rr, cc] = 200
    
    # Retângulos
    imagem[200:250, 400:450] = 50  # Retângulo escuro
    imagem[50:80, 200:260] = 220   # Retângulo claro
    
    # Triângulo
    triangle_rows = [150, 190, 190]
    triangle_cols = [500, 480, 520]
    rr, cc = draw.polygon(triangle_rows, triangle_cols)
    imagem[rr, cc] = 180
    
    # Estrela
    star_rows = [350, 360, 370, 360, 350, 340, 330, 340]
    star_cols = [100, 110, 120, 130, 140, 130, 120, 110]
    rr, cc = draw.polygon(star_rows, star_cols)
    imagem[rr, cc] = 90
    
    # Adicionar algum ruído
    ruido = np.random.normal(0, 10, imagem.shape)
    imagem = np.clip(imagem + ruido, 0, 255)
    
    # Aplicar um leve blur para suavizar
    imagem = filters.gaussian(imagem, sigma=1)
    
    # Garantir que está no range 0-255 e é uint8
    imagem = np.clip(imagem, 0, 255).astype(np.uint8)
    
    return imagem

def criar_imagem_natural_com_objetos():
    """Cria uma imagem que simula objetos naturais"""
    imagem = np.ones((450, 650)) * 120
    
    # Simular folhas (círculos com variações)
    folha_centros = [(80, 100), (120, 300), (200, 180), (280, 400), (350, 250)]
    for i, (cy, cx) in enumerate(folha_centros):
        raio = 20 + i*2
        rr, cc = draw.disk((cy, cx), raio)
        valid_mask = (rr < imagem.shape[0]) & (cc < imagem.shape[1])
        intensidade = 70 + i*15
        imagem[rr[valid_mask], cc[valid_mask]] = intensidade
        
        # Adicionar um pequeno caule
        if i % 2 == 0:
            imagem[cy:cy+15, cx-2:cx+2] = 40
    
    # Padrão de flores (estrutura mais complexa)
    flor_centros = [(150, 500), (300, 100), (400, 550)]
    for cy, cx in flor_centros:
        # Centro da flor
        rr, cc = draw.disk((cy, cx), 8)
        valid_mask = (rr < imagem.shape[0]) & (cc < imagem.shape[1])
        imagem[rr[valid_mask], cc[valid_mask]] = 30
        
        # Pétalas
        for angulo in range(0, 360, 60):
            rad = np.radians(angulo)
            petala_y = int(cy + 15 * np.sin(rad))
            petala_x = int(cx + 15 * np.cos(rad))
            rr, cc = draw.disk((petala_y, petala_x), 6)
            valid_mask = (rr < imagem.shape[0]) & (cc < imagem.shape[1])
            imagem[rr[valid_mask], cc[valid_mask]] = 200
    
    # Adicionar textura de fundo
    textura = np.random.normal(0, 5, imagem.shape)
    imagem = np.clip(imagem + textura, 0, 255)
    
    # Aplicar blur suave
    imagem = filters.gaussian(imagem, sigma=0.8)
    
    # Garantir que está no range 0-255 e é uint8
    imagem = np.clip(imagem, 0, 255).astype(np.uint8)
    
    return imagem

def extrair_template_da_imagem(imagem, coordenadas, tamanho):
    """Extrai um template de uma região específica da imagem"""
    y, x = coordenadas
    h, w = tamanho
    
    # Garantir que as coordenadas estão dentro dos limites
    y = max(0, min(y, imagem.shape[0] - h))
    x = max(0, min(x, imagem.shape[1] - w))
    
    template = imagem[y:y+h, x:x+w].copy()
    return template

def salvar_imagens_para_upload(imagem_principal, template, nome_base="teste"):
    """Salva as imagens e prepara para upload"""
    # Criar diretório temporário
    if not os.path.exists('template_test_images'):
        os.makedirs('template_test_images')
    
    # Verificar e converter imagens para uint8 se necessário
    if imagem_principal.dtype != np.uint8:
        imagem_principal = np.clip(imagem_principal, 0, 255).astype(np.uint8)
    
    if template.dtype != np.uint8:
        template = np.clip(template, 0, 255).astype(np.uint8)
    
    # Salvar imagem principal
    caminho_imagem = f'template_test_images/{nome_base}_principal.png'
    imsave(caminho_imagem, imagem_principal)
    
    # Salvar template
    caminho_template = f'template_test_images/{nome_base}_template.png'
    imsave(caminho_template, template)
    
    print(f"Imagens salvas em: {caminho_imagem} e {caminho_template}")
    
    # Verificar se as imagens foram salvas corretamente
    if os.path.exists(caminho_imagem) and os.path.exists(caminho_template):
        print("✅ Imagens salvas com sucesso!")
        
        # Verificar tamanhos dos arquivos
        tamanho_principal = os.path.getsize(caminho_imagem)
        tamanho_template = os.path.getsize(caminho_template)
        print(f"📁 Tamanho do arquivo principal: {tamanho_principal} bytes")
        print(f"📁 Tamanho do arquivo template: {tamanho_template} bytes")
    else:
        print("❌ Erro ao salvar imagens!")
    
    return caminho_imagem, caminho_template

def visualizar_imagens_geradas(imagem_principal, template, titulo="Imagens Geradas"):
    """Visualiza as imagens geradas"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Imagem principal
    axes[0,0].imshow(imagem_principal, cmap='gray')
    axes[0,0].set_title('Imagem Principal')
    axes[0,0].set_axis_off()
    
    # Template
    axes[0,1].imshow(template, cmap='gray')
    axes[0,1].set_title('Template para Busca')
    axes[0,1].set_axis_off()
    
    # Imagem principal com região do template destacada
    axes[0,2].imshow(imagem_principal, cmap='gray')
    
    # Encontrar coordenadas do template na imagem principal
    y, x = 70, 70  # Posição onde o template foi extraído
    h, w = template.shape
    
    # Desenhar retângulo ao redor da região do template
    rect = plt.Rectangle((x, y), w, h, edgecolor='red', 
                        facecolor='none', linewidth=3)
    axes[0,2].add_patch(rect)
    axes[0,2].set_title('Imagem com Template Destacado')
    axes[0,2].set_axis_off()
    
    # Histograma da imagem principal
    axes[1,0].hist(imagem_principal.ravel(), bins=50, alpha=0.7, color='blue')
    axes[1,0].set_title('Histograma - Imagem Principal')
    axes[1,0].set_xlabel('Intensidade')
    axes[1,0].set_ylabel('Frequência')
    axes[1,0].grid(True, alpha=0.3)
    
    # Histograma do template
    axes[1,1].hist(template.ravel(), bins=50, alpha=0.7, color='green')
    axes[1,1].set_title('Histograma - Template')
    axes[1,1].set_xlabel('Intensidade')
    axes[1,1].set_ylabel('Frequência')
    axes[1,1].grid(True, alpha=0.3)
    
    # Informações das imagens
    axes[1,2].axis('off')
    info_text = f"""
    INFORMAÇÕES DAS IMAGENS:
    
    IMAGEM PRINCIPAL:
    - Dimensões: {imagem_principal.shape}
    - Tipo: {imagem_principal.dtype}
    - Intensidade min: {imagem_principal.min()}
    - Intensidade max: {imagem_principal.max()}
    - Intensidade média: {imagem_principal.mean():.1f}
    
    TEMPLATE:
    - Dimensões: {template.shape}
    - Tipo: {template.dtype}
    - Intensidade min: {template.min()}
    - Intensidade max: {template.max()}
    - Intensidade média: {template.mean():.1f}
    """
    axes[1,2].text(0.1, 0.9, info_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def gerar_conjunto_imagens_geometricas():
    """Gera conjunto 1: Objetos geométricos"""
    print("🟦 Gerando conjunto de imagens geométricas...")
    imagem = criar_imagem_com_objetos_geometricos()
    
    # Extrair template (um dos círculos)
    template = extrair_template_da_imagem(imagem, (70, 70), (60, 60))
    
    print(f"📊 Estatísticas da imagem principal: {imagem.shape}, {imagem.dtype}, range: [{imagem.min()}, {imagem.max()}]")
    print(f"📊 Estatísticas do template: {template.shape}, {template.dtype}, range: [{template.min()}, {template.max()}]")
    
    visualizar_imagens_geradas(imagem, template, "Conjunto Geométrico")
    
    # Salvar imagens
    caminhos = salvar_imagens_para_upload(imagem, template, "geometrico")
    
    return imagem, template, caminhos

def gerar_conjunto_imagens_naturais():
    """Gera conjunto 2: Objetos naturais"""
    print("🌿 Gerando conjunto de imagens naturais...")
    imagem = criar_imagem_natural_com_objetos()
    
    # Extrair template (uma das folhas)
    template = extrair_template_da_imagem(imagem, (75, 95), (50, 50))
    
    print(f"📊 Estatísticas da imagem principal: {imagem.shape}, {imagem.dtype}, range: [{imagem.min()}, {imagem.max()}]")
    print(f"📊 Estatísticas do template: {template.shape}, {template.dtype}, range: [{template.min()}, {template.max()}]")
    
    visualizar_imagens_geradas(imagem, template, "Conjunto Natural")
    
    # Salvar imagens
    caminhos = salvar_imagens_para_upload(imagem, template, "natural")
    
    return imagem, template, caminhos

def menu_gerador_imagens():
    """Menu interativo para gerar imagens de teste"""
    print("\n" + "="*60)
    print("🎨 GERADOR DE IMAGENS PARA TESTE - TEMPLATE MATCHING")
    print("="*60)
    print("\nEscolha o tipo de imagens para gerar:")
    print("1 - 🟦 Objetos Geométricos (círculos, retângulos, triângulos)")
    print("2 - 🌿 Objetos Naturais (folhas, flores)")
    print("3 - 🎯 Gerar TODOS os conjuntos")
    
    opcao = input("\nDigite sua escolha (1-3): ").strip()
    
    if opcao == '1':
        return gerar_conjunto_imagens_geometricas()
    elif opcao == '2':
        return gerar_conjunto_imagens_naturais()
    elif opcao == '3':
        print("\n🔄 Gerando todos os conjuntos de imagens...")
        resultados = []
        print("\n" + "="*40)
        print("🟦 GERANDO OBJETOS GEOMÉTRICOS")
        print("="*40)
        resultados.append(gerar_conjunto_imagens_geometricas())
        
        print("\n" + "="*40)
        print("🌿 GERANDO OBJETOS NATURAIS")
        print("="*40)
        resultados.append(gerar_conjunto_imagens_naturais())
        
        print("\n🎉 TODOS OS CONJUNTOS FORAM GERADOS COM SUCESSO!")
        return resultados
    else:
        print("❌ Opção inválida. Gerando conjunto geométrico por padrão...")
        return gerar_conjunto_imagens_geometricas()

# Executar o gerador
if __name__ == "__main__":
    print("🚀 INICIANDO GERADOR DE IMAGENS DE TESTE")
    print("💡 As imagens serão salvas na pasta 'template_test_images/'")
    
    resultado = menu_gerador_imagens()
    
    print("\n" + "="*60)
    print("✅ GERADOR CONCLUÍDO!")
    print("="*60)
    print("\n📁 As imagens foram salvas na pasta 'template_test_images/'")
    print("\n🎯 AGORA USE A APLICAÇÃO DE TEMPLATE MATCHING:")
    print("   1. Execute a célula do Template Matching Avançado")
    print("   2. Escolha a opção 2: 'Análise com imagens da pasta template_test_images'")
    print("   3. As imagens serão detectadas automaticamente!")
    print("\n📊 Você verá análises detalhadas com:")
    print("   • Histogramas de intensidade")
    print("   • Mapas de correlação")
    print("   • Estatísticas avançadas")
    print("   • Métricas de qualidade")