import numpy as np
import pandas as pd
from pprint import pprint


def viterbi(pi, a, b, obs):

    # Tamanho da matriz de emissão (oculto-observável)
    num_estados = np.shape(b)[0]
    T = np.shape(obs)[0]
    print(num_estados)

    # Inicia o caminho em branco, do tamanho da sequência de observação.
    caminho = np.zeros(T, dtype=int)

    delta = np.zeros((num_estados, T))
    phi = np.zeros((num_estados, T))

    # Inicializa delta e phi.
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    # Segue a sequência de observações.
    print('\nSeguindo...\n')

    for t in range(1, T):
        for s in range(num_estados):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s}, t={t}: phi[{s}, {t}] = {phi}'.format(
                s=s, t=t, phi=phi[s, t]))

    # Encontra o caminho ótimo.
    print('-'*100)
    print("Backtracking:\n")
    caminho[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        caminho[t] = phi[caminho[t+1], [t+1]]
        print("caminho[{}] = {}".format(t, caminho[t]))
    print()

    return caminho, delta, phi

###################
# Estados ocultos #
###################


# Estados ocultos
oculto_estados = ['Quente', 'Frio']
# Probabilidades iniciais
pi = [0.3, 0.7]
# Transições entre estados ocultos
oculto_transicoes = [[0.7, 0.3], [0.4, 0.6]]

oculto_espaco = pd.Series(pi, index=oculto_estados, name="Estados Ocultos")
a_df = pd.DataFrame(columns=oculto_estados, index=oculto_estados)
for i, transicao in enumerate(oculto_transicoes):
    a_df.loc[oculto_estados[i]] = transicao

# Matriz A
a = a_df.values

#######################
# Estados observáveis #
#######################

# Estados observáveis
observaveis_estados = ["Ensolarado", "Chuvoso", "Nublado"]
# Transições oculto para observável
oculto_obs_transicoes = [[0.2, 0.6, 0.2], [0.4, 0.1, 0.5]]
b_df = pd.DataFrame(columns=observaveis_estados, index=oculto_estados)
for i, transicao in enumerate(oculto_obs_transicoes):
    b_df.loc[oculto_estados[i]] = transicao
print(b_df, end="\n\n")

# Matriz B
b = b_df.values

######################
# Sequência de ações #
######################

# Mapa dos estados observáveis.
obs_mapa = {'Ensolarado': 0, 'Chuvoso': 1, 'Nublado': 2}

# Sequência de estados observados.
sequencia = list(input("Sequencia de observacoes: "))
sequencia = [int(i) for i in sequencia]
obs = np.array(sequencia)
# obs = np.array([1, 1, 2, 1, 0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1])

oculto_obs_mapa = dict((v, k) for k, v in obs_mapa.items())
obs_seq = [oculto_obs_mapa[v] for v in list(obs)]
print(pd.DataFrame(np.column_stack([obs, obs_seq]),
                   columns=['obs', 'obs_seq']))

# Algoritmo de Viterbi
path, delta, phi = viterbi(pi, a, b, obs)
# print("\n Caminho de estados ótimo: \n", path)
# print("\n delta: \n", delta)
# print("\n phi: \n", phi)

estados_mapa = {0: 'Frio', 1: 'Quente'}
estados_caminho = [estados_mapa[v] for v in path]

print(pd.DataFrame()
      .assign(observacao=obs_seq)
      .assign(caminho_otimo=estados_caminho))
