# StockPrediction
The goal of this project is to use a RNN model to predict the price of airline based on previous price and volume. 

## stock.py
This program makes the models, this current version focuses on some large
airlines:

* NAS.OL      Norwegian Air Shuttle ASA  
* RYAA        Ryanair Holdings plc        
* LHA.DE      Deutsche Lufthansa AG       
* IAG.MC      International Consolidated Airlines Group   
* AF.PA       Air France-KLM SA          
* EZJ.L       easyJet plc      

This is becasue we want the model to learn about the spesicifics of airline stocks in case there are any subtleties.


## predict2.py
This program takes the model form stock.py and makes a prediciton to your Airline of choice.
