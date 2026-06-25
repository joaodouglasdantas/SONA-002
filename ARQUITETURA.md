# SONA — Photo → The Sims 3 (Arquitetura)

Módulo do app SONA que recebe a **foto de um rosto humano real** e devolve um
**retrato no estilo de The Sims 3**, preservando a identidade da pessoa.

---

## 1. Decisão central: onde a IA roda

O alvo é **mobile**. Os modelos necessários (SDXL + InstantID, ou Flux) exigem
**8–16 GB+ de VRAM** e centenas de MB a GB de pesos. Isso **não roda em celular**.

> Decisão: **app fino no celular → backend faz toda a IA**.
> O celular só captura/envia a foto e exibe o resultado.

```
┌─────────────┐   HTTPS multipart    ┌──────────────────────────────────────┐
│  App mobile │ ───────────────────► │            Backend (FastAPI)          │
│ (iOS/Andr.) │                      │                                       │
│  captura    │ ◄─────────────────── │  1. Pré-processo  2. Validação rosto  │
│   exibe     │   PNG + metadados    │  3. Geração (IA)  4. Verif. fidelidade│
└─────────────┘                      └───────────────┬───────────────────────┘
                                                     │ chamada ao provider
                                          ┌──────────▼───────────┐
                                          │  Provider de geração │
                                          │  Replicate (InstantID)│
                                          │  ou ComfyUI self-host │
                                          └──────────────────────┘
```

**Provider abstrato** (`generation/base.py`): hoje usamos **Replicate** (InstantID
hospedado, sem precisar de GPU própria, pago por imagem). Quando o volume crescer,
troca-se para **ComfyUI self-hosted** numa GPU sem alterar o resto do código —
só a implementação do provider. Isso evita lock-in e otimiza custo conforme escala.

---

## 2. Pipeline (gatekeeper + geração + verificação)

```
foto recebida
   │
   ▼
[A] Pré-processamento ── corrige EXIF, normaliza formato, redimensiona, valida tamanho/MIME
   │
   ▼
[B] VALIDAÇÃO DE ROSTO (porteiro — rejeita aqui se falhar)
   ├─ B1 Detecção facial (InsightFace / SCRFD)   → precisa de ≥1 rosto, score e tamanho mínimos
   ├─ B2 "É rosto humano REAL?" (CLIP zero-shot)  → rejeita desenho, cartoon, 3D/render, anime,
   │                                                animal, objeto, paisagem
   ├─ B3 Estratégia de múltiplos rostos           → escolhe o rosto PRINCIPAL (ou rejeita se ambíguo)
   └─ B4 Embedding de identidade (ArcFace 512-d)   → guarda assinatura facial p/ condicionar e verificar
   │
   ▼
[C] GERAÇÃO (InstantID sobre SDXL)
   ├─ Identidade: embedding ArcFace → IdentityNet do InstantID
   ├─ Pose/enquadramento: keypoints faciais da foto original (ControlNet do InstantID)
   └─ Estilo: prompt "The Sims 3" + (opcional) LoRA de estilo + negative prompt
   │
   ▼
[D] VERIFICAÇÃO DE FIDELIDADE (qualidade)
   ├─ re-detecta rosto na saída
   ├─ similaridade de cosseno ArcFace(entrada) × ArcFace(saída)
   └─ abaixo do limiar → re-tenta com parâmetros ajustados (até N vezes) ou marca baixa fidelidade
   │
   ▼
resultado + metadados (similaridade, rosto usado, tentativas)
```

---

## 3. Justificativa das tecnologias

### Detecção e identidade — **InsightFace** (SCRFD + ArcFace)
Padrão da indústria em reconhecimento facial. SCRFD dá detecção robusta com
landmarks; ArcFace dá um **embedding de 512 dimensões** que é o que melhor mede
"é a mesma pessoa?". Esse mesmo embedding serve para **duas coisas**: condicionar
o InstantID (preservar identidade) e **verificar a fidelidade** no final.

### "Rosto humano real" — **CLIP zero-shot** (`B2`)
O requisito mais delicado: detectores faciais **também disparam em cartoons,
renders 3D e personagens de jogo**. Detecção sozinha não basta. Usamos um
classificador **zero-shot com CLIP**: comparamos a imagem contra rótulos como
*"a real photograph of a human face"* vs *"a drawing / 3D render / video game
character / cartoon / animal"*. Se a foto não for classificada como foto real
de humano com confiança suficiente, **rejeita**. Não precisa treinar nada e
cobre desenho, ilustração, personagem de jogo, animal, objeto e paisagem.
(É plugável: dá para trocar por um modelo anti-spoofing/liveness depois.)

### Múltiplos rostos — **rosto principal** (`B3`)
Estratégia padrão: escolhe o rosto **principal** = maior área de bounding box
ponderada pelo score de detecção e pela centralidade. Se houver dois rostos de
tamanho parecido (ambíguo), **rejeita e pede outra foto** — assim nunca
"misturamos" pessoas nem geramos a pessoa errada. Tudo configurável.

### Geração — **InstantID sobre SDXL** (e não LoRA por pessoa)
- **InstantID** preserva identidade a partir de **uma única foto**, sem treino.
  Injeta o embedding facial via IdentityNet e mantém **pose/enquadramento** pelos
  keypoints (ControlNet). É o melhor custo-benefício de fidelidade para retrato.
- **LoRA por pessoa** daria fidelidade altíssima, mas exige **várias fotos da
  mesma pessoa + treino por usuário** — inviável para foto única num app. ❌
- **IP-Adapter FaceID** é uma alternativa/fallback; InstantID costuma preservar
  melhor identidade + pose em retratos. Mantemos como caminho secundário.

### Estilo The Sims 3 — **prompt engineering + LoRA opcional**
O visual Sims 3 = render 3D estilizado, pele levemente "plástica"/CG, sombreamento
suave, olhos característicos, estética Maxis. Conseguimos isso com **prompt + negative
prompt** fortes; para fidelidade de estilo ainda maior, dá para plugar uma **LoRA
de estilo "The Sims 3"** (Civitai) via parâmetro, sem mudar código.

### API — **FastAPI**
Assíncrona, validação por Pydantic, tipagem, ótima para upload multipart e para
ser consumida por um cliente mobile. Imports pesados (torch/insightface/clip) são
**preguiçosos** para o servidor subir leve e os testes rodarem sem GPU.

---

## 4. Como os requisitos do projeto são atendidos

| Requisito | Onde é tratado |
|---|---|
| Usuário envia foto | endpoint `POST /transform` (multipart) |
| Verificação de rosto antes de tudo | etapa `[B]` (porteiro) |
| Aceitar só rosto humano real | `B1` detecção + `B2` CLIP real-vs-não-real |
| Rejeitar desenho/jogo/animal/objeto/paisagem | `B2` |
| Estratégia p/ múltiplos rostos | `B3` (rosto principal, ou rejeita se ambíguo) |
| Preservar traços/identidade | embedding ArcFace → InstantID (`B4`+`C`) |
| Preservar expressão/pose/enquadramento | keypoints faciais → ControlNet do InstantID |
| Só mudar o estilo artístico | prompt/LoRA Sims 3; identidade travada pelo InstantID |
| Não criar personagem aleatório / não mudar identidade | verificação de fidelidade `[D]` com limiar |
| Não mudar gênero/idade/traços marcantes | prompt neutro + InstantID + checagem de similaridade |
| Ignorar tudo que não for rosto na validação | validação opera só sobre o rosto detectado |

---

## 5. Estrutura de pastas

```
SONA 002/
├─ app/
│  ├─ config.py              # configurações (limiares, modelo, chaves) via env
│  ├─ main.py                # FastAPI: /transform, /health
│  ├─ schemas.py             # modelos Pydantic de resposta
│  ├─ pipeline.py            # orquestrador A→B→C→D
│  ├─ face/
│  │  ├─ detector.py         # InsightFace: detecção + embedding ArcFace
│  │  ├─ realness.py         # CLIP zero-shot: rosto humano real?
│  │  └─ validator.py        # porteiro: combina detector+realness+multi-rosto
│  ├─ generation/
│  │  ├─ base.py             # interface Provider + dataclass de request
│  │  ├─ replicate_provider.py
│  │  ├─ comfyui_provider.py # caminho self-hosted (stub documentado)
│  │  └─ prompts.py          # engenharia de prompt The Sims 3
│  ├─ verify/similarity.py   # cosseno ArcFace entrada×saída
│  └─ utils/image_io.py      # decode, EXIF, resize
├─ web/index.html            # UI de teste (envia foto, mostra antes/depois)
├─ tests/                    # testes com provider/IA mockados
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## 6. Limites e próximos passos
- A geração real exige **chave do provider** (Replicate) ou **GPU** (ComfyUL).
  O código está pronto e modular; basta configurar `.env`.
- Evoluções: cache de resultados, fila de jobs (Celery/RQ) para escala, modelo
  anti-spoofing dedicado em `B2`, e LoRA própria de The Sims 3 fine-tunada.
