## Project Question ##

**What is the question you hope to answer?**

This is a question devised by: [Bentley et al](http://www.peterjbentley.com/heartchallenge/).

Using digital recordings of heartbeats, can we (1) classify the sound into its respective (a) lub and (b) dub actions, and (2) can we classify the heartbeat as normal or caused by a heart condition (specifically murmur, extra heart sound, artifact).

**What data are you planning to use to answer that question?**

The Data is available as a [Kaggle dataset] (https://www.kaggle.com/kinguistics/heartbeat-sounds).

**What do you know about the data so far?**

The data comes with two main datasets. Each dataset will provide audio files of heartbeats and a text files defining each audio file and labels for classification.
* Dataset A:
  * Dataset A is a set of recordings that were made by the general public using an Iphone. 
  * It also comes with a text file defining the heartbeat cycle vs time
  * It contains four categories of conditions:
    * Normal
    * Murmur
    * Extra Heart Sound
    * Artifact
* Dataset B:
  * Dataset B is a set of recordings that were collected in a hospital using a digital stethascope
  * It contains three categories of conditions:
    * Normal
    * Murmur
    * Extrasystole

**Why did you choose this topic?**

Seems like an interesting challenge to identify the real/important features from a noisy signal. Also applying these techniques in the medical industry is always very useful because it can decrease a lot of subjective determination and variance across medical professionals.
