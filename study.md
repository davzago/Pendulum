La DQN fa l'update dei pesi dopo ogni step, visto che il dataset ha shape (batch, time, data_spec) per calcolare la loss prende due traiettorie visto che time=2 e la usa per fare l'update

La rete riesce a muovere il pendolo fino alla posizione verticale ma una volta raggiunta smette di compiere azioni in quanto il reward è una funzione dell'angolo e dell'azione compiuta, il massimo reward è 0 che si ottiene quando il pendolo è verticale e non si compie nessuna azione. Questo probabilmente succede perchè l'agente è troppo greedy e non riesce a vedere che non fare nulla lo porta a perdere di più nel lungo periodo, per risolvere questo problema ci possono essere diverse soluzioni:

- aumentare il numero di step per calcolare la loss (Time)
- aumentare il numero di step prima di fare l'update della rete target (fatto) 
- modificare il parametro di discount gamma (è già ad 1) (fatto)
- dare più esperienza all'agente nella situazione di pendolo verticale nel replay buffer
- ingrandire la rete
- usare nodi RNN
- cambiare la funzione reward:
  - togliere la componente correlata alla azione
  - dare dei punti extra quando il pendolo si trova in posizione verticale

Il training viene fatto usando next(iterator) come esperienze, quello che esce da questa operazione è un tensore [64,2,traj.shape], questo tensore viene usato per calcolare la loss e quindi i pesi della rete principale vengono aggiornati a ogni batch, la target network viene invece aggiornata solamente dopo 20 (tareget_update_period) trainig step ovvero dopo aver analizzato un totale di 20 batch