# Q-Learning con reti neurali

Il Q-learning è una procedura che ha come obbiettivo far imparare ad un'agente la migliore azione da compiere in un determinato stato dell'ambiente in cui si trova, questo normalmente viene fatto riempendo una tabella contenente un Q-value per ogni transizione tra stati.
La regola utilizzata per aggiornare il Q value è: 
$$
Q(s,a) = r + \gamma max_{a'}Q(s',a')
$$
dove $r$ è il reward determinato dall'azione compiuta e $\gamma$ è il fattore di discount che decide l'importanza delle azioni successivo sul Q-value attuale. 
L'agente esplorando l'ambiente e utilizzando questa regola riempirà la tabella fino a convergere alla migliore funzione $Q^*$.
Una policy è una "regola" che l'agente segue per decidere che azioni compiere nel suo ambiente, nel caso del Q-Learning la policy che l'agente segue è quella di scegliere l'azione con corrispondente Q-value maggiore.
Tuttavia questo metodo non può essere applicato se gli stati raggiungibili sono non numerabili, per ovviare a questo problema si può provare ad usare una rete neurale per cercare di approssimare la funzione Q*. Il limite di questo metodo è che il numero di azioni che si possono compiere in ogni stato deve essere finito.

## Ambiente 

L'ambiente in esame è **Pendulum-v0** di gym, in questo ambiente l'obbiettivo è far stare in piedi un pendolo spingendolo a destra e a sinistra con forza variabile. Il reward dato a ogni time step è una funzione dell'angolo del pendolo e della spinta data al pendolo e il suo valore massimo è 0, questo accade quando il pendolo è in verticale e non viene applicata nessuna forza.
Per poter utilizzare una rete neurale in questo caso è stato necessario discretizzare il le azioni disponibili in quanto il numero di azioni deve essere pari al numero di neuroni di output della rete, quindi l'azione che poteva essere scelta in un intervallo continuo [-2,2] diventa una scelta tra [-2, 1, 0, 1, 2] per un totale di 5 possibili azioni per ogni stato.  

## DQN

Una Deep Q Network è una rete che prende in input lo stato e l'azione compiuta e restituisce il Q-value corrispondente, lo stato è definito dall'oggetto chiamato observation e contiene 3 valori: il seno e il coseno dell'angolo del pendolo e la velocità angolare corrente.
La rete dunque è costituita da un input layer di 4 nodi (observation + azione) seguito da 2 hidden layers di 100 e 50 nodi rispettivamente con *ReLu* come funzione di attivazione e infine un output layer con 5 nodi (1 per ogni azione) con attivazione lineare.

## Loss Function

Per effettuare il Training di questa rete c'è bisogno di una loss function, visto che stiamo applicando il Q-Learning vogliamo che l'output della rete diventi un Q-value, per ottenere questo risultato si utilizza:
$$
L(\theta)=E[(r + \gamma max_{a'} Q(s',a';\theta)- Q(s,a;\theta))^2]
$$
dove $Q(s,a;\theta)$ è il valore stimato dalla rete e i valori (s,a,s',a') sono estratti dal replay buffer (nostro training set), il valore di $\gamma$ in questo caso è $0.9$, in questo modo gli stati e le azioni future hanno un peso abbastanza alto ma non hanno lo stesso peso dello stato attuale.

Una cosa da notare è che per calcolare la loss function c'è bisogno di utilizzare 2 time step, inoltre la rete viene utilizzata sia per stimare il Q-value sia per stimare il valore del target, il che è un problema in quanto facciamo un update della rete anche il target cambia. Nella pratica questo non portava buoni risultati, per questo motivo ho utilizzato un'altra rete (target) strutturalmente identica a quella principale ma che viene aggiornata ogni 20 training steps, questo fa si che il target non cambi in continuazione e rende il training più stabile.  

## Replay buffer

Il replay buffer è il training set che verrà usato per allenare la DQN, visto che la rete ha bisogno di due time step per calcolare la loss a ogni train step alla rete viene fornito un Tensore di forma [64, 2, 4] dove 64 è la dimensione del batch, 2 è il numero di time step che servono a calcolare la loss function e 4 è la dimensione di un input.
La collezione dei dati viene effettuata sia prima che in contemporanea al training, inizialmente vengono collezionati una serie di esempi con azioni casuali, dopo ogni training step vengono collezionati dei dati aggiuntivi utilizzando la policy di collezione dell'agente che in pratica utilizza i q-value forniti dalla rete ma con una certa probabilità $\epsilon = 0.1$ compie una azione casuale, questo permette all'agente di esplorare l'ambiente e trovare stati e azioni migliori.

## Risultati

Per dare un valore ai risultati ogni 1000 training step l'agente è stato valutato in 10 episodi di 200 time step ciascuno, da questi episodi si ottiene il reward medio, il miglore risultato è -254 ottenuto dopo la fine della fase di training.

<img src="/home/davide/Documenti/Pendulum/avg_return_graph.jpg" alt="avg_return" style="zoom:67%;" />

Dopo il training l'agente risulta in grado di mettere in verticale il pendolo molto velocemente e di mantenerlo nella stessa posizione, il reward medio sarà comunque sempre negativo in quanto il massimo reward è 0 e compiere azioni porta reward negativi.