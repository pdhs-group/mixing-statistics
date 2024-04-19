# -*- coding: utf-8 -*-
"""
Hallo und herzlich willkommen zu unserem kleinen Praktikum. In diesem Skript möchte ich ihnen die Grundzüge von Python näherbringen. Wichtig vorab: Die Inhalte in Python sind ergänzend gedacht und als solche natürlich freiwillig. Ich werde Sie in der Prüfung nichts Python-bezogenes fragen.

Es folgt keine umfassende Einführung in Python. Hierzu nutzen Sie die vielen ausgezeichneten Ressourcen die Online zu finden sind. Hier einige Tips für den Start
https://realpython.com/
https://www.learnpython.org/
https://www.google.de/ (;D)
"""

# Ich bin ein Kommentar und führe Sie durch das Skript. 
# Python ist eine Interpretersprache und nicht wie z. B. C++ eine Compilersprache. Das heißt, wenn Sie ein Skript ausführen wird Python dieses Zeile für Zeile ausführen und nicht erst den gesamten Code in Maschinencode überführen (compilieren). 
# Die Python Syntax ist sehr eingängig und ähnelt stark der von MATLAB. Einige wesentliche Unterschiede gibt es jedoch, auf die wir hier nicht näher eingehen wollen.

# Starten wir klassischerweise mit einer Textausgabe
print('Hello World <3')

# Python ist open-source und basiert auf sogenannten packages. Diese werden von vielen Menschen weltweit aktiv weiterentwickelt und die Chancen sind hoch, dass Sie das passende package für nahezu jedes Problem finden. Es folgen die wesentlichen, häufig verwendeten packages, die mit dem import Befehl eingebunden werden. >>as "XYZ"<< ist hier eine Hilfestellung, um nicht jedes Mal den vollen Namen ausschreiben zu müssen. 
import numpy as np # Generelle Funktionalität und Matrixoperationen
import matplotlib.pyplot as plt # Plots und Visualisierung
import scipy # Viele wissenschaftliche und statistische Tools

# Ich bin eine "list", die alle möglichen Dinge speichert, auf die wieder zugegriffen werden kann.
# Zwischen '' steht ein String, 12 ist ein Integer, 3.14 ein Float und True ein Boolean. In Python muss der Typ einer Variable nicht zuvor definiert werden!
l = ['hello','world',12,3.14,True] 
print('Erster Eintrag der Liste: ', l[0])
print('Vierter Eintrag der Liste: ', l[3])

# Für mathematische Zwecke bieten sich numpy arrays an, die beliebig viele Dimensionen besitzen können (N-D Arrays). So können wir z.B. eine 2D Matrix der Größe (2,3) mit 1 initialisieren und einzelne Elemente austauschen
M = np.ones((2,3))
M[1,2] = 0
print(M)

# Hiermit lässt sich auch wunderbar Rechnen. Der Operator "**" bedeutet eine Potenz
M = M/2
M = M**2
print(M)

# Gehen wir nur alle Zeilen und Reihen von M durch und erhöhen die Werte um 1. Das geht elegant, oder wie hier gezeigt mit eine Schleife. range(n) startet (falls nicht anders angegeben) bei 0 und zählt bis n (Achtung: n selbst ist nicht enthalten!). Wir erhalten die Längen der Matrix M über den Befehl M.shape. Index 0 beschreibt die Zeilen, Index 1 die Spalten 
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        M[i,j] = M[i,j] + 1

# Besonderheit: Pyhton berücksichtigt Einrückungen und gibt folglich auch Fehler aus wenn logisch gruppierte Inhalte (z.B. Inhalte einer Schleife) nicht auf der selben Breite eingerückt sind. 
print(M)

# Wir können auch unterfunktionen definieren, die wir mehrmals brauchen. Das erspart Schreib-Arbeit!
def do_the_thing(in_put):
    out_put = in_put + 1
    return out_put

M[0,0] = do_the_thing(M[0,0])
M[0,0] = do_the_thing(M[0,0])
print(M)

# Abschließend noch ein einfaches Beispiel zum Plotten von Daten. Wir generieren einen 1D-Array aus x-Werten mit identischem Abstand, berechnen daraus y und plotten das Ganze anschließend.
x = np.linspace(0,100,100)
y = x*2
plt.plot(x,y,color='b',label='tolle Daten')
plt.legend()
plt.savefig('export/hello_world_testplot.pdf')
