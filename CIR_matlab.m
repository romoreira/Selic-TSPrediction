% Passo 1: Importar os dados do CSV
T = readtable('selicdados2.csv');
T.Data = datetime(T.Data, 'InputFormat', 'dd/MM/yyyy'); % Converte a coluna 'Data' para o formato de data
data = table2array(T(:, 2)); % extrair apenas a coluna 'SelicDia'
delta = 1/12; % definir o intervalo de tempo

% Separar os dados em treino e teste
% Calcular o índice que separa os conjuntos de treino e teste
total_samples = size(data, 1);
train_size = floor(0.8 * total_samples); % 80% para treino
test_size = total_samples - train_size;

% Dividir os dados em conjuntos de treino e teste
dataTrain = data(1:train_size, :);
dataTest = data(train_size+1:end, :);


% Plotar a série temporal dos dados de treino
figure;
plot(1:train_size, dataTrain, 'b', 'LineWidth', 1.5); % Por exemplo, usei a cor azul ('b') e uma linha mais grossa (LineWidth)
xlabel('Tempo');
ylabel('Valor');
title('Série Temporal - Conjunto de Treino');



% Plotar a série temporal dos dados de teste
figure;
plot(1:test_size, dataTest, 'r', 'LineWidth', 1.5); % Por exemplo, usei a cor vermelha ('r') e uma linha mais grossa (LineWidth)
xlabel('Tempo');
ylabel('Valor');
title('Série Temporal - Conjunto de Teste');


%size(dataTrain)
%size(dataTest)

% Definir uma função de log-verossimilhança para o modelo CIR
% Estimar os parâmetros do modelo CIR usando mle
params0 = [0.3505492 0.04295122 0.08127823]; % valores iniciais dos parâmetros
lb = [0 0 0]; % limites inferiores dos parâmetros
ub = [Inf Inf Inf]; % limites superiores dos parâmetros
options = statset('MaxIter', 10000); % opções de otimização
params = mle(dataTrain,'logpdf',@loglikecir,'start',params0,'lower',lb,'upper',ub,'options',options);

% Criar um objeto cir com os parâmetros estimados
alpha = params(1); % velocidade de reversão à média
theta = params(2); % nível de reversão à média
sigma = params(3); % volatilidade
CIR = cir(alpha,theta,sigma);

% Simular os caminhos futuros da taxa de juros usando o objeto cir
nPeriods = length(dataTest); % número de períodos de previsão
nTrials = 100000; % número de simulações
x0 = dataTrain(end,1); % valor inicial da taxa de juros
rng(42); % fixar a semente aleatória para reprodutibilidade
[X,T] = simulate(CIR,nPeriods,'DeltaTime',delta,'NTRIALS',nTrials);

X = X(2:end,:);
T = T(2:end);



% Calcular os erros de todas as simulações
MSE = zeros(nTrials,1); % vetor para armazenar os erros quadráticos médios
MAE = zeros(nTrials,1); % vetor para armazenar os erros absolutos médios
for i = 1:nTrials
    MSE(i) = mean((dataTest - X(:,i)).^2); % calcular o MSE da i-ésima simulação
    MAE(i) = mean(abs(dataTest - X(:,i))); % calcular o MAE da i-ésima simulação
end

% Encontrar a melhor simulação de acordo com o critério escolhido
[~,best] = min(MSE); % encontrar o índice da simulação com o menor MSE
%[~,best] = min(MAE); % encontrar o índice da simulação com o menor MAE

% Extrair o vetor de valores simulados da melhor simulação
YPred = X(:,best);

% Calcular o MSE entre os dados de teste e a melhor simulação
MSE = mean((dataTest - YPred).^2);

% MAE: mean absolute error
MAE = mean(abs(dataTest - YPred));

% RMSE: root mean squared error
RMSE = sqrt(mean((dataTest - YPred).^2));

%X
figure;
plot(dataTest, 'b'); % Plotar dataTest em azul ('b')
hold on; % Manter o gráfico atual e adicionar outro
plot(YPred, 'r'); % Plotar YPred em vermelho ('r')

xlabel('Tempo');
ylabel('Taxa de juros');
title('Comparação entre Dados de Teste e Valores Preditos');
legend('Dados de Teste', 'Valores Preditos');

% Calcular a média e o intervalo de confiança da taxa de juros prevista
meanX = mean(X,2); % média da taxa de juros em cada período
stdX = std(X,0,2); % desvio padrão da taxa de juros em cada período
ciX = [meanX - 1.96*stdX, meanX + 1.96*stdX]; % intervalo de confiança de 95% da taxa de juros em cada período


% Exibindo as métricas no console
fprintf('MSE: %f\n', MSE);
fprintf('MAE: %f\n', MAE);
fprintf('RMSE: %f\n', RMSE);


% Definir uma função de log-verossimilhança para o modelo CIR
function logL = loglikecir(params,data,varargin)
    alpha = params(1); % velocidade de reversão à média
    theta = params(2); % nível de reversão à média
    sigma = params(3); % volatilidade
    delta = 1/12; % intervalo de tempo
    x = data(1:end-1); % taxa de juros observada
    y = data(2:end); % taxa de juros deslocada
    c = 2*alpha/(sigma^2*(1-exp(-alpha*delta)));
    q = 2*alpha*theta/sigma^2-1;
    u = c*exp(-alpha*delta)*x;
    v = c*y;
    logL = -sum(log(c) + 0.5*log(v) + (u+v)/2.*besseli(q,sqrt(u.*v),1) - 0.5*(u+v));
    
    % Verificação para lidar com possíveis argumentos extras
    if ~isempty(varargin)
        % Faça algo com os argumentos extras, se necessário
    end
end
