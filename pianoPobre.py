import time, cv2
import numpy as np
import rtmidi

# Notas MIDI correspondentes às teclas virtuais
NOTAS = [ 60, 62, 64, 65, 67, 69, 71, 72, 74 ] 
VELOCIDADE_NOTA = 127
NOME_JANELA = "PianoPobre v1.0"
ALTURA_TECLA = 0.25

LARGURA_RECONHECEDOR = 500
TAMANHO_KERNEL = 0.042
TEMPO_RESET = 5
TEMPO_VERIFICACAO_SALVAR = 1
LIMIAR = 25
VALOR_COMPARACAO = 128

# Número de teclas e estado inicial (não tocando)
numeroTeclas = len(NOTAS)
tocando = numeroTeclas * [False]

# Configuração de saída MIDI
saidaMidi = rtmidi.MidiOut()
assert(saidaMidi.get_ports())
numeroPorta = 0 if len(saidaMidi.get_ports()) == 1 or 'through' not in str(saidaMidi.get_ports()[0]).lower() else 1
saidaMidi.open_port(numeroPorta)

# Captura de vídeo da webcam
video = cv2.VideoCapture(0)
larguraQuadro = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alturaQuadro = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ajuste de tamanho do quadro de reconhecimento
if LARGURA_RECONHECEDOR >= larguraQuadro:
    larguraEscalada = larguraQuadro
    alturaEscalada = alturaQuadro
else:
    proporcao = larguraQuadro / alturaQuadro
    larguraEscalada = LARGURA_RECONHECEDOR
    alturaEscalada = int(LARGURA_RECONHECEDOR / proporcao)

tamanhoKernel = 2*int(TAMANHO_KERNEL*larguraEscalada/2)+1

# Criação de uma sobreposição em branco
sobreposicaoEmBranco = np.zeros((alturaQuadro, larguraQuadro, 3), dtype=np.uint8)

cv2.namedWindow(NOME_JANELA, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(NOME_JANELA, larguraQuadro, alturaQuadro)

# Cálculo das áreas das teclas na escala e no quadro original
retangulosEscalados = []
retangulosQuadro = []

for i in range(numeroTeclas):
    x0 = larguraEscalada*i//numeroTeclas
    x1 = larguraEscalada*(i+1)//numeroTeclas-1

    r = [(x0, 0), (x1, int(ALTURA_TECLA*alturaEscalada))]
    retangulosEscalados.append(r)

    x0 = larguraQuadro*i//numeroTeclas
    x1 = larguraQuadro*(i+1)//numeroTeclas-1

    r = [(x0, 0), (x1, int(ALTURA_TECLA*alturaQuadro))]
    retangulosQuadro.append(r)

# Limites das teclas escaladas e no quadro
cantosSuperioresEsquerdoEscalado = (min(r[0][0] for r in retangulosEscalados), min(r[0][1] for r in retangulosEscalados))
cantosInferioresDireitoEscalado = (max(r[1][0] for r in retangulosEscalados), max(r[1][1] for r in retangulosEscalados))
cantosSuperioresEsquerdoQuadro = (min(r[0][0] for r in retangulosQuadro), min(r[0][1] for r in retangulosQuadro))
cantosInferioresDireitoQuadro = (max(r[1][0] for r in retangulosQuadro), max(r[1][1] for r in retangulosQuadro))
larguraTeclasEscalada = cantosInferioresDireitoEscalado[0] - cantosSuperioresEsquerdoEscalado[0]
alturaTeclasEscalada = cantosInferioresDireitoEscalado[1] - cantosSuperioresEsquerdoEscalado[1]

# Verifica se as dimensões são válidas
if larguraTeclasEscalada <= 0 or alturaTeclasEscalada <= 0:
    raise ValueError("Dimensões calculadas são inválidas: larguraTeclasEscalada <= 0 ou alturaTeclasEscalada <= 0")

# Criação de uma matriz de teclas
teclas = np.zeros((alturaTeclasEscalada, larguraTeclasEscalada), dtype=np.uint8)

def ajustarParaTeclas(xy):
    return (xy[0]-cantosSuperioresEsquerdoEscalado[0], xy[1]-cantosSuperioresEsquerdoEscalado[1])
    
for i in range(numeroTeclas):
    r = retangulosEscalados[i]
    cv2.rectangle(teclas, ajustarParaTeclas(r[0]), ajustarParaTeclas(r[1]), i+1, cv2.FILLED)

# Variáveis de quadro para comparação
quadroComparacao = None
quadroSalvo = None
tempoSalvo = 0
tempoUltimaVerificacao = 0

def comparar(a, b):
    return cv2.threshold(cv2.absdiff(a, b), LIMIAR, VALOR_COMPARACAO, cv2.THRESH_BINARY)[1]
    
while True:
    ok, quadro = video.read()
    if not ok:
        time.sleep(0.05)
        continue
    quadro = cv2.flip(quadro, 1)

    # Extrai a área das teclas do quadro
    quadroTeclas = quadro[cantosSuperioresEsquerdoQuadro[1]:cantosInferioresDireitoQuadro[1], cantosSuperioresEsquerdoQuadro[0]:cantosInferioresDireitoQuadro[0]]
    if larguraEscalada != larguraQuadro:
        quadroTeclas = cv2.resize(quadroTeclas, (larguraTeclasEscalada, alturaTeclasEscalada))
    quadroTeclas = cv2.cvtColor(quadroTeclas, cv2.COLOR_BGR2GRAY)
    borrado = cv2.GaussianBlur(quadroTeclas, (tamanhoKernel, tamanhoKernel), 0)

    t = time.time()
    salvar = False
    if quadroSalvo is None:
        salvar = True
        tempoUltimaVerificacao = t
    else:
        if t >= tempoUltimaVerificacao + TEMPO_VERIFICACAO_SALVAR:
            if VALOR_COMPARACAO in comparar(quadroSalvo, borrado):
                print("Calculando Interpolações...")
                salvar = True
            tempoUltimaVerificacao = t
        if t >= tempoSalvo + TEMPO_RESET:
            print("Recriando banco de imagens...")
            quadroComparacao = borrado
            salvar = True
    if salvar:
        quadroSalvo = borrado
        tempoSalvo = t
            
    if quadroComparacao is None:
        quadroComparacao = borrado
        continue
        
    # Comparação do quadro atual com o quadro de comparação
    delta = comparar(quadroComparacao, borrado)
    soma = teclas + delta
    
    sobreposicao = sobreposicaoEmBranco.copy()

    for i in range(numeroTeclas):
        r = retangulosQuadro[i]
        if 1+i+VALOR_COMPARACAO in soma:
            cv2.rectangle(sobreposicao, r[0], r[1], (255,255,255), cv2.FILLED)
            if not tocando[i]:
                saidaMidi.send_message([0x90, NOTAS[i], VELOCIDADE_NOTA])
                tocando[i] = True
        else:
            if tocando[i]:
                saidaMidi.send_message([0x80, NOTAS[i], 0])
                tocando[i] = False
        cv2.rectangle(sobreposicao, r[0], r[1], (0,255,0), 2)

    cv2.imshow(NOME_JANELA, cv2.addWeighted(quadro, 1, sobreposicao, 0.25, 1.0))
    
    if (cv2.waitKey(1) & 0xFF) == 27 or cv2.getWindowProperty(NOME_JANELA, 0) == -1:
        break

# Libera os recursos
video.release()
cv2.destroyAllWindows()
del saidaMidi