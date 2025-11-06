
```markdown
# 📚 Documentação Completa - Sistema de Trading com IA

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Sistema](#arquitetura-do-sistema)
3. [Módulos Detalhados](#módulos-detalhados)
4. [Fluxo de Dados](#fluxo-de-dados)
5. [Modelo de Machine Learning](#modelo-de-machine-learning)
6. [Gestão de Risco](#gestão-de-risco)
7. [Indicadores Técnicos](#indicadores-técnicos)
8. [Backtesting](#backtesting)
9. [Paper Trading](#paper-trading)
10. [Live Trading](#live-trading)
11. [Configurações Avançadas](#configurações-avançadas)
12. [Otimização e Tuning](#otimização-e-tuning)
13. [Troubleshooting](#troubleshooting)
14. [FAQ](#faq)
15. [Glossário](#glossário)

---

## Visão Geral

### O que é este sistema?

Este é um sistema completo de trading automatizado que utiliza Deep Learning (LSTM + Attention Mechanism) para analisar o mercado de criptomoedas e executar operações de compra e venda automaticamente na exchange Binance.

### Principais Características

- **Inteligência Artificial**: Modelo LSTM com mecanismo de atenção para análise de padrões
- **Análise Técnica**: Mais de 30 indicadores técnicos calculados automaticamente
- **Gestão de Risco**: Stop Loss, Take Profit e controle de exposição
- **Backtesting**: Teste sua estratégia em dados históricos
- **Paper Trading**: Simule operações em tempo real sem risco
- **Live Trading**: Execute operações reais na Binance
- **Visualizações**: Gráficos detalhados de performance

### Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **PyTorch**: Framework de Deep Learning
- **Pandas/NumPy**: Manipulação de dados
- **TA-Lib**: Indicadores técnicos
- **Binance API**: Integração com a exchange
- **Matplotlib/Seaborn**: Visualizações

---

## Arquitetura do Sistema

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────┐
│                     SISTEMA DE TRADING                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Binance    │─────▶│    Data      │─────▶│  Feature  │ │
│  │     API      │      │  Collector   │      │ Engineer  │ │
│  └──────────────┘      └──────────────┘      └─────┬─────┘ │
│                                                      │       │
│                                                      ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Trading    │◀─────│     LSTM     │◀─────│  Training │ │
│  │    System    │      │    Model     │      │   Data    │ │
│  └──────┬───────┘      └──────────────┘      └───────────┘ │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Backtest    │      │    Paper     │                    │
│  │   Engine     │      │   Trading    │                    │
│  └──────────────┘      └──────────────┘                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Fluxo de Execução

1. **Coleta de Dados**: API Binance → Data Collector
2. **Processamento**: Data Collector → Feature Engineer
3. **Análise**: Features → Modelo LSTM
4. **Decisão**: Modelo → Trading System
5. **Execução**: Trading System → Binance API
6. **Monitoramento**: Trading System → Métricas/Logs

---

## Módulos Detalhados

### 1. config.py

**Propósito**: Centraliza todas as configurações do sistema.

#### Parâmetros Principais

##### API Binance
```python
BINANCE_API_KEY = "sua_api_key"
BINANCE_API_SECRET = "sua_api_secret"
```
- Credenciais para acessar a API da Binance
- Obtenha em: Binance → API Management

##### Parâmetros de Trading
```python
SYMBOL = "BTCUSDT"          # Par de trading
TIMEFRAME = "15m"           # Intervalo das velas
INITIAL_CAPITAL = 1000      # Capital inicial em USDT
```

**Timeframes disponíveis**:
- `1m`: 1 minuto (alta frequência, mais ruído)
- `5m`: 5 minutos (frequência média)
- `15m`: 15 minutos (RECOMENDADO - melhor equilíbrio)
- `30m`: 30 minutos (conservador)
- `1h`: 1 hora (swing trading)
- `4h`: 4 horas (posições mais longas)
- `1d`: 1 dia (investimento)

##### Gestão de Risco
```python
RISK_PER_TRADE = 0.02       # 2% de risco por trade
STOP_LOSS_PCT = 0.015       # 1.5% stop loss
TAKE_PROFIT_PCT = 0.03      # 3% take profit
MAX_POSITIONS = 3           # Máximo de posições simultâneas
```

**Explicação**:
- `RISK_PER_TRADE`: Percentual do capital que você está disposto a perder em um trade
- `STOP_LOSS_PCT`: Distância do stop loss em relação ao preço de entrada
- `TAKE_PROFIT_PCT`: Distância do take profit (objetivo de lucro)
- `MAX_POSITIONS`: Limita exposição total ao mercado

##### Parâmetros do Modelo
```python
SEQUENCE_LENGTH = 60        # Número de velas para análise
HIDDEN_SIZE = 128          # Tamanho da camada oculta
NUM_LAYERS = 2             # Número de camadas LSTM
DROPOUT = 0.3              # Taxa de dropout (regularização)
LEARNING_RATE = 0.001      # Taxa de aprendizado
BATCH_SIZE = 32            # Tamanho do batch
EPOCHS = 100               # Épocas de treinamento
```

**Explicação**:
- `SEQUENCE_LENGTH`: Quantas velas o modelo analisa para fazer uma predição
- `HIDDEN_SIZE`: Capacidade do modelo (maior = mais complexo)
- `NUM_LAYERS`: Profundidade da rede (mais camadas = mais abstração)
- `DROPOUT`: Previne overfitting (0.3 = 30% de neurônios desligados)

##### Divisão de Dados
```python
TRAIN_SPLIT = 0.7          # 70% para treino
VAL_SPLIT = 0.15           # 15% para validação
TEST_SPLIT = 0.15          # 15% para teste
```

##### Backtesting
```python
LOOKBACK_DAYS = 90         # Dias de histórico para análise
```

##### Modo de Operação
```python
MODE = "backtest"          # "backtest", "paper", "live"
```

---

### 2. data_collector.py

**Propósito**: Coleta dados da Binance e gerencia informações da conta.

#### Classe: BinanceDataCollector

##### Métodos Principais

###### `__init__(api_key, api_secret)`
Inicializa o cliente da Binance.

```python
collector = BinanceDataCollector(api_key, api_secret)
```

###### `get_historical_data(symbol, interval, lookback_days)`
Coleta dados históricos de velas.

**Parâmetros**:
- `symbol`: Par de trading (ex: "BTCUSDT")
- `interval`: Timeframe (ex: "15m")
- `lookback_days`: Quantos dias de histórico

**Retorna**: DataFrame com colunas:
- `timestamp`: Data/hora da vela
- `open`: Preço de abertura
- `high`: Preço máximo
- `low`: Preço mínimo
- `close`: Preço de fechamento
- `volume`: Volume negociado

**Exemplo**:
```python
df = collector.get_historical_data("BTCUSDT", "15m", 90)
print(f"Coletadas {len(df)} velas")
```

###### `get_realtime_data(symbol, interval, limit=100)`
Coleta dados em tempo real.

**Parâmetros**:
- `symbol`: Par de trading
- `interval`: Timeframe
- `limit`: Número de velas recentes

**Uso**: Paper trading e live trading

###### `get_account_balance()`
Retorna saldo da conta Binance.

**Retorna**: Dicionário com saldos por ativo:
```python
{
    'USDT': {'free': 1000.0, 'locked': 0.0, 'total': 1000.0},
    'BTC': {'free': 0.05, 'locked': 0.0, 'total': 0.05}
}
```

###### `get_current_price(symbol)`
Retorna preço atual de um par.

**Exemplo**:
```python
price = collector.get_current_price("BTCUSDT")
print(f"Preço atual do BTC: ${price:.2f}")
```

#### Tratamento de Erros

O módulo trata automaticamente:
- Erros de conexão
- Rate limits da API
- Dados inválidos
- Timeout de requisições

---

### 3. feature_engineering.py

**Propósito**: Cria indicadores técnicos a partir dos dados brutos.

#### Classe: FeatureEngineer

##### Método Principal: `create_features(df)`

Recebe um DataFrame com OHLCV e adiciona 30+ features técnicas.

#### Indicadores Criados

##### 1. Médias Móveis

**Médias Móveis Simples (SMA)**:
```python
sma_9   # Média de 9 períodos (curto prazo)
sma_21  # Média de 21 períodos (médio prazo)
sma_50  # Média de 50 períodos (longo prazo)
```

**Médias Móveis Exponenciais (EMA)**:
```python
ema_9   # EMA de 9 períodos
ema_21  # EMA de 21 períodos
ema_50  # EMA de 50 períodos
```

**Interpretação**:
- Preço acima da média = tendência de alta
- Preço abaixo da média = tendência de baixa
- Cruzamento de médias = sinal de mudança de tendência

##### 2. RSI (Relative Strength Index)

```python
rsi  # RSI de 14 períodos
```

**Interpretação**:
- RSI > 70: Sobrecomprado (possível queda)
- RSI < 30: Sobrevendido (possível alta)
- RSI = 50: Neutro

##### 3. MACD (Moving Average Convergence Divergence)

```python
macd         # Linha MACD
macd_signal  # Linha de sinal
macd_diff    # Histograma (diferença)
```

**Interpretação**:
- MACD cruza acima do sinal: Sinal de compra
- MACD cruza abaixo do sinal: Sinal de venda
- Histograma positivo: Momentum de alta
- Histograma negativo: Momentum de baixa

##### 4. Bollinger Bands

```python
bb_high      # Banda superior
bb_mid       # Banda média (SMA 20)
bb_low       # Banda inferior
bb_width     # Largura das bandas (volatilidade)
bb_position  # Posição do preço nas bandas (0-1)
```

**Interpretação**:
- Preço toca banda superior: Possível reversão de baixa
- Preço toca banda inferior: Possível reversão de alta
- Bandas estreitas: Baixa volatilidade (possível breakout)
- Bandas largas: Alta volatilidade

##### 5. ATR (Average True Range)

```python
atr  # ATR de 14 períodos
```

**Interpretação**:
- Mede volatilidade do ativo
- ATR alto: Mercado volátil
- ATR baixo: Mercado calmo
- Usado para ajustar stops

##### 6. Stochastic Oscillator

```python
stoch_k  # Linha %K
stoch_d  # Linha %D (sinal)
```

**Interpretação**:
- > 80: Sobrecomprado
- < 20: Sobrevendido
- Cruzamento: Sinal de entrada/saída

##### 7. Volume

```python
volume_sma    # Média de volume
volume_ratio  # Razão volume atual / média
```

**Interpretação**:
- Volume alto + alta: Confirmação de tendência
- Volume alto + queda: Possível reversão
- Volume baixo: Falta de convicção

##### 8. VWAP (Volume Weighted Average Price)

```python
vwap  # Preço médio ponderado por volume
```

**Interpretação**:
- Preço acima VWAP: Compradores no controle
- Preço abaixo VWAP: Vendedores no controle

##### 9. Momentum

```python
momentum  # Diferença de preço em 10 períodos
roc       # Rate of Change (%)
```

**Interpretação**:
- Momentum positivo: Força compradora
- Momentum negativo: Força vendedora

##### 10. Price Action

```python
price_change      # Variação percentual
high_low_ratio    # Amplitude da vela
close_open_ratio  # Corpo da vela
```

##### 11. Padrões de Candlestick

```python
body           # Tamanho do corpo
upper_shadow   # Sombra superior
lower_shadow   # Sombra inferior
```

**Interpretação**:
- Corpo grande: Forte movimento
- Sombras longas: Indecisão
- Corpo pequeno: Consolidação

##### 12. Tendência

```python
trend_sma  # Tendência por SMA (1 = alta, -1 = baixa)
trend_ema  # Tendência por EMA (1 = alta, -1 = baixa)
```

#### Método: `get_feature_columns(df)`

Retorna lista de colunas que são features (exclui OHLCV e timestamp).

**Exemplo**:
```python
engineer = FeatureEngineer()
df = engineer.create_features(df)
features = engineer.get_feature_columns(df)
print(f"Total de features: {len(features)}")
```

---

### 4. model.py

**Propósito**: Define e treina o modelo de Deep Learning.

#### Arquitetura do Modelo

##### Classe: TradingLSTM

```
Input (features)
    ↓
LSTM Layer 1 (128 units)
    ↓
LSTM Layer 2 (128 units)
    ↓
Multi-Head Attention (4 heads)
    ↓
Fully Connected 1 (64 units) + BatchNorm + ReLU + Dropout
    ↓
Fully Connected 2 (32 units) + BatchNorm + ReLU + Dropout
    ↓
Fully Connected 3 (3 units)
    ↓
Output (BUY, SELL, HOLD)
```

##### Componentes

**1. LSTM (Long Short-Term Memory)**
- Processa sequências temporais
- Captura dependências de longo prazo
- Mantém memória de padrões passados

**2. Attention Mechanism**
- Foca nas partes mais importantes da sequência
- Melhora a capacidade de decisão
- 4 cabeças de atenção para múltiplas perspectivas

**3. Batch Normalization**
- Estabiliza o treinamento
- Acelera convergência
- Reduz overfitting

**4. Dropout**
- Regularização para prevenir overfitting
- Desliga aleatoriamente 30% dos neurônios
- Força o modelo a aprender features robustas

##### Saídas do Modelo

O modelo produz 3 probabilidades:
- **Classe 0 (BUY)**: Sinal de compra
- **Classe 1 (SELL)**: Sinal de venda
- **Classe 2 (HOLD)**: Manter posição

#### Classe: TradingModelTrainer

##### Método: `prepare_data(df, feature_columns)`

Prepara dados para treinamento:

1. **Normalização**: StandardScaler para features
2. **Criação de Labels**: Baseado em retornos futuros
   - Retorno > 1%: BUY (0)
   - Retorno < -1%: SELL (1)
   - Caso contrário: HOLD (2)
3. **Criação de Sequências**: Janelas deslizantes de tamanho `SEQUENCE_LENGTH`

**Exemplo**:
```python
trainer = TradingModelTrainer(config)
sequences, labels = trainer.prepare_data(df, feature_columns)
```

##### Método: `train(train_loader, val_loader, input_size)`

Treina o modelo:

**Processo**:
1. Inicializa modelo e otimizador (Adam)
2. Define loss function (CrossEntropyLoss)
3. Loop de treinamento:
   - Forward pass
   - Calcula loss
   - Backward pass
   - Atualiza pesos
4. Validação a cada época
5. Early stopping se não melhorar por 10 épocas
6. Salva melhor modelo

**Métricas monitoradas**:
- Train Loss
- Train Accuracy
- Validation Loss
- Validation Accuracy

**Exemplo de saída**:
```
Epoch [5/100]
  Train Loss: 0.8234 | Train Acc: 65.23%
  Val Loss: 0.8567 | Val Acc: 63.45%
```

##### Método: `predict(sequence)`

Faz predição para uma sequência:

**Retorna**:
- `prediction`: Classe predita (0, 1, ou 2)
- `confidence`: Confiança da predição (0-1)

**Exemplo**:
```python
prediction, confidence = trainer.predict(sequence)
if prediction == 0 and confidence > 0.7:
    print("Sinal de COMPRA com alta confiança!")
```

##### Método: `save_model(path)` e `load_model(path)`

Salva/carrega modelo treinado e scaler.

**Arquivos gerados**:
- `trading_model.pth`: Pesos do modelo
- `scaler.pkl`: Scaler para normalização

---

### 5. trading_system.py

**Propósito**: Gerencia posições, risco e execução de ordens.

#### Classe: Position

Representa uma posição aberta ou fechada.

**Atributos**:
```python
symbol          # Par de trading
type            # 'LONG' ou 'SHORT'
entry_price     # Preço de entrada
size            # Quantidade
stop_loss       # Preço do stop loss
take_profit     # Preço do take profit
entry_time      # Timestamp de entrada
exit_price      # Preço de saída (quando fechada)
exit_time       # Timestamp de saída
pnl             # Profit & Loss
status          # 'OPEN' ou 'CLOSED'
close_reason    # 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL', 'MANUAL'
```

#### Classe: TradingSystem

##### Método: `calculate_position_size(entry_price)`

Calcula tamanho ideal da posição baseado em risco.

**Fórmula**:
```
Valor em Risco = Capital × RISK_PER_TRADE
Distância Stop = Entry Price × STOP_LOSS_PCT
Tamanho = Valor em Risco / Distância Stop
```

**Limitações**:
- Máximo 20% do capital por posição
- Arredondamento para precisão da Binance

**Exemplo**:
```
Capital: $1000
Risk per trade: 2% = $20
Entry: $50,000
Stop Loss: 1.5% = $750
Tamanho: $20 / $750 = 0.0267 BTC
```

##### Método: `open_position(signal, price, timestamp)`

Abre nova posição.

**Processo**:
1. Verifica se pode abrir (MAX_POSITIONS)
2. Calcula tamanho da posição
3. Define stop loss e take profit
4. Executa ordem (se live/paper)
5. Adiciona à lista de posições

**Retorna**: Objeto Position ou None

##### Método: `close_position(position, price, timestamp, reason)`

Fecha posição existente.

**Processo**:
1. Calcula P&L
2. Executa ordem de fechamento (se live/paper)
3. Atualiza capital
4. Registra no histórico
5. Atualiza estatísticas

**Cálculo de P&L**:
- **LONG**: (Exit Price - Entry Price) × Size
- **SHORT**: (Entry Price - Exit Price) × Size

##### Método: `check_stops(current_price, timestamp)`

Verifica stop loss e take profit de todas as posições abertas.

**Lógica**:
- **LONG**:
  - Stop Loss: current_price ≤ stop_loss
  - Take Profit: current_price ≥ take_profit
- **SHORT**:
  - Stop Loss: current_price ≥ stop_loss
  - Take Profit: current_price ≤ take_profit

**Execução**: Automática a cada iteração

##### Método: `update_equity(current_price, timestamp)`

Atualiza equity considerando P&L não realizado.

**Fórmula**:
```
Equity = Capital + Unrealized P&L
```

**Unrealized P&L**: Lucro/prejuízo de posições abertas

##### Método: `get_statistics()`

Calcula métricas de performance.

**Métricas retornadas**:
- `total_trades`: Total de operações
- `winning_trades`: Operações lucrativas
- `losing_trades`: Operações com prejuízo
- `win_rate`: Taxa de acerto (%)
- `total_pnl`: P&L total ($)
- `total_return`: Retorno total (%)
- `avg_win`: Ganho médio por trade vencedor
- `avg_loss`: Perda média por trade perdedor
- `profit_factor`: Razão ganho/perda
- `sharpe_ratio`: Retorno ajustado ao risco
- `max_drawdown`: Maior queda do capital (%)

**Fórmulas**:

**Win Rate**:
```
Win Rate = (Winning Trades / Total Trades) × 100
```

**Profit Factor**:
```
Profit Factor = (Avg Win × Winning Trades) / |Avg Loss × Losing Trades|
```

**Sharpe Ratio**:
```
Sharpe = (Mean Return / Std Return) × √252
```

**Max Drawdown**:
```
Drawdown = (Current Equity - Peak Equity) / Peak Equity × 100
Max Drawdown = Min(Drawdown)
```

##### Método: `_execute_order(position, action)`

Executa ordem na Binance (apenas live trading).

**Modos**:
- **backtest**: Não executa (simulação)
- **paper**: Simula execução
- **live**: Executa ordem real

**Tipos de ordem**:
- Market Buy/Sell para LONG
- Futures para SHORT (requer conta margin)

---

### 6. backtest.py

**Propósito**: Testa estratégia em dados históricos.

#### Classe: Backtester

##### Método: `run_backtest(model_trainer, df, feature_columns)`

Executa backtest completo.

**Processo**:
1. Inicializa TradingSystem em modo backtest
2. Itera sobre dados históricos
3. Para cada vela:
   - Cria sequência de features
   - Faz predição com o modelo
   - Verifica stops
   - Executa ação (se confiança > 60%)
   - Atualiza equity
4. Fecha posições abertas no final
5. Retorna TradingSystem com resultados

**Lógica de Decisão**:
```python
if confidence > 0.6:
    if prediction == 0:  # BUY
        open_position(LONG)
    elif prediction == 1:  # SELL
        close_long_positions()
```

##### Método: `plot_results(trading_system, df)`

Gera 3 gráficos:

**1. Preço e Trades**
- Linha do preço
- Marcadores de entrada (▲ verde = LONG, ▼ vermelho = SHORT)
- Marcadores de saída (● = lucro, ✕ = prejuízo)

**2. Curva de Equity**
- Linha de equity ao longo do tempo
- Área verde: Acima do capital inicial
- Área vermelha: Abaixo do capital inicial

**3. Drawdown**
- Gráfico de área mostrando quedas do capital
- Identifica períodos de perda

**Salva**: `backtest_results.png`

##### Método: `plot_trade_analysis(trading_system)`

Gera 4 gráficos analíticos:

**1. P&L por Trade**
- Barras verdes: Trades lucrativos
- Barras vermelhas: Trades com prejuízo

**2. Histograma de P&L**
- Distribuição dos resultados
- Identifica assimetria

**3. P&L Cumulativo**
- Evolução do lucro ao longo dos trades
- Mostra consistência

**4. Razões de Fechamento**
- Pizza chart com:
  - TAKE_PROFIT: Objetivo atingido
  - STOP_LOSS: Stop acionado
  - SIGNAL: Fechado por sinal contrário
  - MANUAL: Fechado manualmente

**Salva**: `trade_analysis.png`

---

### 7. main.py

**Propósito**: Interface principal do sistema.

#### Função: `train_model()`

Executa pipeline completo de treinamento:

**Etapas**:
1. **Coleta de dados**: Binance API
2. **Feature engineering**: Cria indicadores
3. **Preparação**: Normaliza e cria sequências
4. **Treinamento**: Treina modelo LSTM
5. **Avaliação**: Testa em conjunto de teste

**Saída**:
- Modelo treinado salvo
- Acurácia no teste
- Objetos trainer, df, fe
