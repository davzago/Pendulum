# Q-Learning con reti neurali

Il Q-learning è una procedura per far imparare ad un'agente la migliore azione da compiere in un determinato stato dell'ambiente in cui si trova, questo normalmente viene fatto riempendo una tabella contenente un Q-value per ogni transizione tra stati.
La regola utilizzata per aggiornare il Q value è: 
$$
Q(s,a) = r + \gamma max_{a'}Q(s',a')
$$
dove $r$ è il reward determinato dall'azione compiuta e $\gamma$ è il fattore di discount che decide l'importanza delle azioni successivo sul Q-value attuale. 
L'agente esplorando l'ambiente e utilizzando questa regola riempirà la tabella fino a convergere alla effettiva funzione $Q^*$.
Una policy è una "regola" che l'agente segue per decidere che azioni compiere nel suo ambiente, nel caso del Q-Learning la policy che l'agente segue è quella di scegliere l'azione con corrispondente Q-value maggiore.
Tuttavia questo metodo non può essere applicato se gli stati raggiungibili sono non numerabili, per ovviare a questo problema si può provare ad usare una rete neurale per cercare di approssimare la funzione Q*.

## DDQN

## LOSS

## Come effettivamente viene fatto il training

## Replay buffer

### policy per la collezione

## Problemi e (possibili) soluzioni

## Risultati con (avg return)

