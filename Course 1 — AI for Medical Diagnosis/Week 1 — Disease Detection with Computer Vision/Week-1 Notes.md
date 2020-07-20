☀ 1. [Welcome to the Specialization with Andrew and Pranav]()<br>

☀ 2. [Demo]()<br>

☀ 3. [Recommended prerequisites]()<br>

☀ 4. [Medical Image Diagnosis]()<br>

☀ 5. [Eye Disease and Cancer Diagnosis]()<br>

☀ 6. [Building and Training a Model for Medical Diagnosis]()<br>

☀ 7. [Training, prediction, and loss]()<br>

☀ 8. [Image Classification and Class Imbalance]()<br>

☀ 9. [Binary Cross Entropy Loss Function]()<br>

☀ 10. [Impact of Class Imbalance on Loss Calculation]()<br>

☀ 11. [Resampling to Achieve Balanced Classes]()<br>

☀ 12. [Multi-Task]()<br>

☀ 13. [Multi-task Loss, Dataset size, and CNN Architectures]()<br>

☀ 14. [Working with a Small Training Set]()<br>

☀ 15. [Generating More Samples]()<br>

☀ 16. [Model Testing]()<br>

☀ 17. [Splitting data by patient]()<br>

☀ 18. [Sampling]()<br>

☀ 19. [Ground Truth and Consensus Voting]()<br>

☀ 20. [Additional Medical Testing]()<br><br>

# 1. Welcome to the Specialization with Andrew and Pranav

If you've completed the deep learning specialization, or the
machine learning course, and you're looking for application areas
to deepen mastery of AI, this is a good specialization to take.<br>

One of the most important things for becoming really good at ML is to
gain practice applying ML to multiple use cases.<br>

This specialization will step you through multiple use cases spanning some of the most
important applications of AI to medicine.<br>

E.g. given an image of a chest X-ray (unstructured image data), can you train a neural
network to diagnose whether or not a patient has pneumonia?<br>

Or, given structured data, such as the patient's lab results, can you train a decision
tree to estimate the risk of heart attack?<br><br>

By working on these concrete problems, you also see a lot of the practical aspects of ML, from
how to deal with imbalanced data sets, to how to work with missing data, to picking the right
evaluation metric.<br>

In ML, we often default to classification accuracy as the metric, but for many applications
it's not the right metric.<br>
So how do you choose a more appropriate one?<br><br>

Even if your current work is not in medicine, you'll find the application scenarios and their practice
really useful, and maybe the specialization will convince you to get more interested in medicine.<br>

If you are interested in medicine, then this is a great specialization to take.<br>
Maybe you can be the one to invent someone that will save a lot of people's lives.<br><br>

This is a 3-course specialization and in the first course, you'll learn about building ML models
for diagnosis.<br>
**Diagnosis** is about identifying disease.<br>

In the first course, you'll build an algorithm that will look at a chest X-ray and determine whether
it contains disease.<br>

You'll also build another algorithm that will look at brain MRIs and identify the location of tumors
in those brain MRIs.<br><br>

While the first course is on diagnosis or identifying disease, the second course will be on predicting
the future health of patients, which is called **prognosis**<br>

In the second course, you'll learn how to work with structured data.<br>
So let's say you have a patient's lab values and their demographics, and
use those to predict the risk of an event such as their risk of death or
heart attack.<br><br>

Finally, in the third course, you'll learn about AI for **treatment**, that is
for the process of medical care and also for information extraction, getting
information out of medical texts<br>

In course-3, you'll learn how to use ML models to be able to estimate what the
effect of a particular treatment would be on a patient<br>

You'll also learn about the application of AI to text for particular tasks like
question-answering and for extracting labels from radiology reports.<br><br>

In this first course of the `AI for Medicine` specialization, you'll learn about
the applications of AI for medical diagnosis.<br>

**Diagnosis** means the `process of determining which disease or condition explains the
person's symptoms, signs, and medical results.`<br>

In particular, you'll be learning how to build and evaluate deep learning models
for the detection of disease from medical images.<br><br>

In the first week, you build a DL model that can interpret chest X-rays to classify
different disease causes<br>

In the second week, you'll implement evaluation methodologies to assess the quality
of your model.<br>

In the third week, you use image segmentation to identify the location and boundaries of
brain tumors in MRI scans.<br><br>

***

# 2. Demo

In this course, just in the first week you'll build a DL model that can interpret chest
X-rays to classify different diseases.<br>

Let's see a demo of what your model will be able to do.<br>

Here we have a chest X-ray.<br>
**Chest X-ray** is the most commonly performed imaging exam in the world
and used to diagnose and treat a variety of diseases.<br>

This is a chest X-ray of a patient who has fluid in their lungs<br>

When you train your model in assignment 1, it will be able to pick up this
abnormality.<br>

Let's see it in action.<br><br>

Here, I've just run this image of the chest X-ray to a DL model on my phone<br>

The model processes this chest X-ray and recognizes that this patient's X-ray is
abnormal<br>

More specifically, when I scroll down it's found that the `patient likely has fluid
in their lungs.`<br>
This condition is called **edema**<br><br>

In course-3, you'll learn about how you can generate these heat maps that show where
in the image the model is finding evidence of the disease<br>

In this course, you'll learn about and apply all the core concepts to build such a model
using real data<br><br>

***

# 3. Recommended prerequisites

For this course and the rest of the courses in this specialization, you won't need any
background in medicine.<br>

There are however, 3 prerequisites that I suggest you feel comfortable with before you take
this course and the other courses in this specialization.<br>

1. The course assumes that you know the **basics of deep learning**<br>
   E.g. you should know the basics of supervised learning, CNNs, and loss functions

2. You should be fairly comfortable being able to **write code in Python**, as you'll be
   using it to process data and build ML models for the assignments for all 3 courses

3. The lecture concepts will assume **some knowledge of probability**<br>
   For instance, you should be able to recognize when we say probability of A given B
   that this is a `conditional probability`<br>

With these 3 prerequisites, you should be good to go.<br><br>

***

# 4. Medical Image Diagnosis

This week we'll be diving straight into building a DL model for the task of
chest X-ray classification.<br>

Many of the ideas that you'll learn through this example are broadly applicable across
many medical imaging tests.<br>

This week, we'll start by looking at 3 examples of medical diagnostic tasks where DL
has achieved incredible performance<br>

We'll then jump into the training procedure for building AI models for medical
imaging<br>

Finally, we'll look at the testing procedure for evaluating the performance
of these models on real data<br><br>

Our first example is in **dermatology** — `the branch of medicine dealing with the skin.`<br>

One of the tasks dermatologists perform is to look at a suspicious region of the skin to
determine whether a mole is skin cancer or not.<br>

Early detection could likely have an enormous impact on skin cancer outcomes<br>

The five-year survival rate of one kind of skin cancer drops significantly if detected in
its later stages<br><br>

In this study, an algorithm is trained to determine whether a region of skin tissue
is cancerous or not<br>

Using hundreds of thousands (129,000) of images and labels as input, a CNN can be
trained to do this task<br>

We'll look at the training of such an algorithm in the course<br>

Once the algorithm has been trained, its predictions can be evaluated against the
predictions of human dermatologists on a new set of images <br>

In this study, it was found that the algorithm performed as well as the dermatologists did.<br>

```
Melanoma — 130 images
Algorithm: AUC = 0.94
```

Don't worry too much about the interpretation of this graph right now<br>

In a future week of the course, we'll look at the evaluation using such a
curve<br>

The main conclusion you'll be able to draw from this graph is that
the algorithm's prediction accuracy was comparable to the prediction
of human dermatologists.<br><br>

***

# 5. Eye Disease and Cancer Diagnosis

Our second example is in **opthalmology** which `deals with the diagnosis
and treatment of eye disorders.`<br>

One well-known study in 2016 looked at **retinal fundus** images, which
photographed the back of the eye<br>

One disease or *pathology* to look at here is **diabetic retinopathy** (DR),
which is `damage to the retina caused by diabetes` and is a major cause of
blindness.<br>

Currently, detecting DR is a time-consuming and manual process that requires
a trained clinician to examine these photos.<br><br>

In this study, an algorithm was developed to determine whether patients had DR
by looking at such photos <br>

This study used over 128,000 images of which only 30% had DR<br>

We'll look at this **data imbalance problem** which is prominent in medicine and
many other fields with real-world data<br>
We'll see some methods for tackling this challenge<br><br>

Similar to the previous study, this study showed that the performance of the
resulting algorithm was comparable to ophthalmologists<br>

In this study, a majority vote of multiple ophthalmologists was used to set
the **reference standard or ground truth**, which is a group of experts'
best guess of a right answer<br>

Later this week in the course, we'll look at how ground truth can be set in such
medical AI studies<br><br>

Our third example is in **histopathology**, a `medical specialty involving the
examination of tissues under the microscope`<br>

One of the tasks that pathologists do is look at scanned microscopic images of
tissue, called **whole-slide images**, and determine the extent to which a cancer has spread<br>

This is important to help plan treatment, predict the course of the disease, and the
chance of recovery<br><br>

In one study in 2017, using only 270 whole-slide images, AI algorithms were developed
and then evaluated against pathologies<br>

It was found that the best algorithms performed as well as the pathologists did<br>

Now in histopathology, the images are very large and cannot be fed directly into an
algorithm without breaking them down<br>

The general setup of these studies is that instead of feeding in one large, high-resolution
digital image of the slide, `several patches are extracted at a high magnification` and used to
train a model<br>

These patches are labeled with the original label of the whole-slide image and then fed into
a DL algorithm<br>

In this way, the algorithm can be trained on hundreds of thousands (here 500,000+) of patches <br><br>

In this course, you'll apply a similar idea of breaking down a large image into smaller images for
model training to the task of **brain tumor segmentation**<br><br>

***

# 6. Building and Training a Model for Medical Diagnosis

Now that you've seen some of the cutting-edge applications of DL to
medical image classification problems, we'll look at how you can
build your own DL model for the medical imaging task of using chest
X-rays to detect multiple diseases with a single model<br>

We'll walk through the process of training a model for chest X-ray
interpretation and look at the key challenges that you'll face in this
process, and how you can go about successfully tackling them<br><br>

We'll start by looking at the task of chest X-ray interpretation<br>

The **chest X-ray** is one of the most common diagnostic imaging procedures
in medicine with about 2 billion chest X-rays that are taken for a year<br>

Chest X-ray interpretation is critical for the detection of many diseases
including pneumonia and lung cancer, which affect millions of people worldwide
each year<br>

Now a radiologist who is trained in the interpretation of chest X-rays looks
at the chest X-ray, looking at the lungs, the heart, and other regions to look
for clues that might suggest if a patient has pneumonia or lung cancer or another
condition<br><br>

Let's look at one abnormality called a **mass** looks like<br>

And I'm not going to first define what a mass is but let's look at 3 chest
X-rays that contain a mass and 3 chest X-rays that are normal<br>

I can then show you a new chest X-ray here and ask you to identify whether
there is a mass<br>

You might be able to correctly identify that this chest X-ray contains a mass,
and it might look similar to things that you see in images containing a mass we saw before,
but not similar to anything in the *normal* images <br>

The way you're learning is very similar to how we're going to teach an algorithm to
detect mass<br><br>

For our own reference, a **mass** is defined as a <code>*lesion* (damage of tissue) seen on a
chest X-ray as greater than 3 centimeters in diameter</code><br>

Let's see how we can train our algorithm to identify masses<br><br>

***

# 7. Training, prediction, and loss

During training, an algorithm is shown images of chest X-rays labeled with
whether they contain a mass or not<br>

The algorithm learns using these images and labels<br>

The algorithm eventually learns to go from a chest X-ray input to produce the
output of whether the X-ray contains a mass<br>

This algorithm can go by different names — you may have heard of the terms `deep learning algorithm`, or
`model`, `neural network`, or `convolutional neural network`<br><br>

The algorithm produces an output in the form of **scores**, which are `probabilities
that the image contains a mass`<br>

[Image in slide contains 2 images with scores of 0.48 and 0.51 each]<br>

When training has not started, these scores, these probability outputs are not
going to match the desired label<br>

Let's say the desired label for mass is 1, and for normal is 0<br>
As obvious, 0.48 is far off from 1 and 0.51 is far off from the desired label of 0<br><br>

We can measure this error by computing a **loss function** — `measures the error between our
output probability and the desired label`<br>

```
0.48 score has Error (Loss) of 0.32
0.51 score has Error (Loss) of 0.31
```

We'll look at how this loss is computed soon enough<br>

Then, a new set of images and desired labels is presented to the algorithm as it learns
to produce scores that are closer to the desired labels over time<br>

```
0.48 score is now 0.60, Error = 0.22
0.51 score is now 0.30, Error = 0.15
```
<br>

***

# 8. Image Classification and Class Imbalance

Typically in the medical AI examples we've seen in previous videos, hundreds of thousands
of images are shown to the algorithm<br>

This is a typical setup for **image classification**, which is a core task in the computer
vision field where a natural image is input to an image classification algorithm, which says
what is the object contained in the image<br>

You may have seen DL algorithms that can do this<br><br>

Our example of chest X-ray classification is similar in many ways to the image classification
setup<br>

There are a few additional challenges which make training medical image classification algorithms
more challenging, which we'll cover next<br>

We'll talk about `3 key challenges` for training algorithms on medical images —<br>
* The Class Imbalance challenge
* The Multi-Task challenge
* The Dataset Size challenge<br>

For each challenge, we'll cover one or two techniques to tackle them<br>

* The Class Imbalance challenge — solved by Weighted Loss / Resampling
* The Multi-Task challenge — solved by Multi-Label Loss
* The Dataset Size challenge — solved by Transfer Learning + Data Augmentation<br><br>


Let's start with the **class imbalance challenge** — There's not an equal number of
examples of `non-disease` and `disease` in medical datasets<br>

This is a reflection of the **prevalence**, or the frequency of disease in the
real-world, while we see that there are a lot more examples of `normal`s than of `mass`,
especially if we're looking at X-rays of a healthy population<br>

In a medical dataset, you might see 100 times as many normal examples as mass examples<br><br>

***

# 9. Binary Cross Entropy Loss Function

This creates a problem for the learning algorithm as they would see mostly normal examples<br>

This yields a model that starts to predict a very low probability of disease for everybody and
won't be able to identify when an example has a disease<br>

Let's see how we can trace this problem to the loss function that we use to train the algorithm<br>

We'll also see how we can modify this loss function in the presence of imbalanced data<br><br>

```
L(X, y) = —log P(Y = 1|X)  if y = 1
L(X, y) = —log P(Y = 0|X)  if y = 0
```
This loss over here is called the **binary cross-entropy loss** and this `measures the performance
of a classification model whose output is between 0 and 1`<br>

Let's look at an example to see how this loss function evaluates<br><br>

Here we have an example of a chest X-ray that contains a mass, so it gets labeled with 1
and the algorithm outputs a probability of 0.2<br>

0.2 is the probability according to the algorithm of Y being equal to 1, i.e.
the probability that this example is a mass (`P(Y = 1|X) = 0.2`)<br>

So now, we can apply the loss function to compute the loss on this example<br>

```
L = —log(0.2) (here y = 1)
```

This evaluates to 0.70 — the loss the algorithm gets on this particular example<br><br>

Let's look at another example, this time a non-mass example which would have a label of 0<br>
Our algorithm outputs a probability of 0.7<br>

```
P(Y = 0|X) = 1 - P(Y = 1|X)

L = —log(1 - 0.7) (here y = 0)
```

This expression evaluates to 0.52<br><br>

***

# 10. Impact of Class Imbalance on Loss Calculation

We've seen how the loss is applied to a single example<br>
Let's see how it applies to a bunch of examples

Examples | Prediction Probabilities | Loss
-------- | ------------------------ | ---
P1 Normal | 0.5 | 0.3
P2 Normal | 0.5 | 0.3
P3 Normal | 0.5 | 0.3
**P4 Mass** | 0.5 | 0.3
P5 Normal | 0.5 | 0.3
P6 Normal | 0.5 | 0.3
**P7 Mass** | 0.5 | 0.3
P8 Normal | 0.5 | 0.3

<br>

Here we have 6 examples that are normal, and 2 that are mass<br>
`P2, P3, P4..` etc. are the Patient IDs<br>

When the training hasn't started, let's say the algorithm produces an output
probability of 0.5 for all of the examples<br>

The loss can then be computed for each of the examples

```
For P1 Normal, —log(1 - 0.5) = —log(0.5) = 0.3
For P4 Mass, —log(0.5) = 0.3
```
<br>

The total contribution to the loss from the `mass` examples = 0.6<br>
Total loss from `normal` examples = 1.8<br>

Notice how most of the contribution to the loss is coming from the normal examples
rather than from the mass examples<br>

So the algorithm is optimizing its updates to get the normal examples right, and not
giving much relative weight to the mass examples <br>

In practice, this doesn't produce a very good classifier — This is the
class imbalance problem<br><br>

The **solution to the class imbalance problem** is to `modify the loss function to
weigh the normal and the mass classes differently`<br>

w<sub>p</sub> will be the weight we assign to the positive or `mass` examples, and<br>
w<sub>n</sub> to the negative or `normal` examples<br>


<code>L(X, y) = w<sub>p</sub> x —log P(Y = 1|X), if y = 1</code><br>
<code>L(X, y) = w<sub>n</sub> x —log P(Y = 0|X), if y = 0</code><br><br>

Let's see what happens when we weigh the positive examples more<br>

We want to weigh the `mass` examples more such that they can have an
equal contribution overall to the loss, as the `normal` examples<br>

Let's pick 6/8 as the weight we have on the `mass` examples and 2/8 as the
weight we have on the `normal` examples <br><br>

Examples | Prediction Probabilities | Loss
-------- | ------------------------ | ---
P1 Normal | 0.5 | (2/8) x 0.3 = 0.075
P2 Normal | 0.5 | (2/8) x 0.3 = 0.075
P3 Normal | 0.5 | (2/8) x 0.3 = 0.075
**P4 Mass** | 0.5 | (6/8) x 0.3 = 0.225
P5 Normal | 0.5 | (2/8) x 0.3 = 0.075
P6 Normal | 0.5 | (2/8) x 0.3 = 0.075
**P7 Mass** | 0.5 | (6/8) x 0.3 = 0.225
P8 Normal | 0.5 | (2/8) x 0.3 = 0.075

<br>

Now, you can see that if you sum up the total loss from the `mass` examples,
we get (0.225 x 2) = 0.45 and this is equal to the total loss from the `normal`
examples (0.075 x 6) = 0.45<br>

In the general case, the weight we'll put on the positive class will be the number
of negative examples over the total number of examples<br>

The weight we'll put on the negative class will be the number of positive examples
over the total number of examples<br>

<code>w<sub>p</sub> = (#negative examples) / (#total examples)</code><br>
<code>w<sub>n</sub> = (#positive examples) / (#total examples)</code><br><br>

With this setting of w<sub>p</sub> and w<sub>n</sub>, we can have over all the
examples, the loss contributions from the positive and negative classes to be the same<br>

So this is the idea of modifying the loss using weights<br>
This method is called the **Weighted Loss** method, used to tackle the class
imbalance problem<br><br>

***

# 11. Resampling to Achieve Balanced Classes

Let's discuss another procedure that we can use to tackle the class imbalance
problem, which is **resampling**<br>

The basic idea here is to resample the dataset such that we have an equal number
of `normal` and `mass` examples<br>

Examples | Re-Sampled
-------- | -----------
P1 Normal | P3 Normal
P2 Normal | P6 Normal
P3 Normal | P1 Normal
**P4 Mass** | P8 Normal
P5 Normal | **P7 Mass**
P6 Normal | **P4 Mass**
**P7 Mass** | **P7 Mass**
P8 Normal | **P4 Mass**

Let's see how we can achieve this<br>

1. We group the `normal` and `mass` examples together <br>
   Notice,, the `normal` group has 6 examples and `mass` has 2 examples

2. Now from these groups, we'll sample the images such that there's an equal
   number of positive and negative samples<br>

   We can do this by sampling half of the examples from the positive or `mass`
   class and half of the examples from the negative or `normal` class<br>

   Note that this means we may not be able to include all of the `normal` examples
   in our re-sample<br>
   Furthermore, we may have more than one copy of the `mass` examples in our
   re-sampled dataset<br>

With this resampled dataset, if we now compute the loss in the same way that we
did before, we can see that even without the weights (using the standard Binary
Cross-Entropy Loss), there's an equal contribution to the loss from the `mass`
examples and from the `normal` examples <br>

Re-Sampled | Prediction Probabilities | Loss
---------- | ------------------------ | ----
P3 Normal | 0.5 | 0.3
P6 Normal | 0.5 | 0.3
P1 Normal | 0.5 | 0.3
P8 Normal | 0.5 | 0.3
**P7 Mass** | 0.5 | **0.3**
**P4 Mass** | 0.5 | **0.3**
**P7 Mass** | 0.5 | **0.3**
**P4 Mass** | 0.5 | **0.3**

There are many variations of this approach like **undersampling** the
`normal` class or **oversampling** the `mass` class<br>

These approaches fall under the category of re-sampling methods,
which we can use to combat the data imbalance problem<br><br>

***

# 12. Multi-Task

Let's chat about another challenge that we encounter in the medical image
classification setting, which is the **multitask** challenge<br>

Thus far, we have looked at binary classification where we care about classifying
whether an example is a `mass` or not a mass (`normal`)<br>

However, in the real-world we care about classifying the presence or absence of many
such diseases<br>

One simple way to do this is to have models that each learn one of these tasks<br><br>

However, maybe we can learn to do all of the tasks using one model<br>

An advantage of this is that we can `learn features that are common to identifying
more than one disease`, allowing us to use our existing data more efficiently<br>

This is the setup of multitask learning<br>

Let's look at how we can train the algorithm to learn all these tasks at the same time<br><br>

So instead of the examples having one label, they now have one label for every disease in the
example where 0 denotes the `absence` of that disease and 1 denotes the `presence` of that disease<br>

Examples (mass, pneumonia, edema) | Prediction Probabilities
--------------------------------  | ------------------------
P1 0, 1, 0 | 0.3, 0.1, 0.8
P2 0, 0, 1 | 0.1, 0.1, 0.8
P3 0, 1, 1 | 0.2, 0.2, 0.7
P4 1, 0, 1 | 0.6, 0.3, 0.8
P5 1, 1, 1 | 0.7, 0.7, 0.9
P6 1, 0, 0 | 0.8, 0.1, 0.2
P7 0, 1, 1 | 0.3, 0.9, 0.8
P8 0, 0, 0 | 0.1, 0.1, 0.2

For the first one, we have an `absence` of mass, `presence` of pneumonia, and an `absence`
of edema, which is excess fluid in the lungs<br>

Instead of having one output from the model, the model now has `3 different outputs` denoting
the probability of three different diseases<br>

To train such an algorithm, we also need to make the modification to the loss function from
the binary task to the multitask setting<br>

Let's look at how we can do that<br><br>

***

# 13. Multi-task Loss, Dataset size, and CNN Architectures

We modify the loss function such that we look at the error associated with each disease<br>

We can represent our new loss as the sum of the losses over the multiple diseases<br>

This is called the **Multi-Label or Multi-Task Loss**<br>

<code> L(X, y) = L(X, y<sub>mass</sub>) + L(X, y<sub>pneumonia</sub>) + L(X, y<sub>edema</sub>)</code><br><br>

In this case, here's the loss that we get for the 8 examples<br>

Examples (mass, pneumonia, edema) | Prediction Probabilities | Loss
--------------------------------  | ------------------------ | ----
P1 0, 1, 0 | 0.3, 0.1, 0.8 | 0.52 + 1.00 + 0.70
P2 0, 0, 1 | 0.1, 0.1, 0.8 | 0.05 + 0.05 + 0.10
P3 0, 1, 1 | 0.2, 0.2, 0.7 | 0.10 + 0.70 + 0.15
P4 1, 0, 1 | 0.6, 0.3, 0.8 | 0.22 + 0.52 + 0.10
P5 1, 1, 1 | 0.7, 0.7, 0.9 | 0.15 + 0.15 + 0.05
P6 1, 0, 0 | 0.8, 0.1, 0.2 | 0.10 + 0.05 + 0.10
P7 0, 1, 1 | 0.3, 0.9, 0.8 | 0.52 + 0.05 + 0.10
P8 0, 0, 0 | 0.1, 0.1, 0.2 | 0.05 + 0.05 + 0.10

For example, in the first row, the first loss term 0.52 represents the loss associated
with the `mass` class using prediction probability = 0.3 and label = 0<br>

In the first row, loss = 1.0 represents the loss associated with the `pneumonia` class using
prediction probability = 0.1 and label = 1<br>

So we add up the three losses given to us by the individual loss function components<br><br>

One final consideration is how we can account for class imbalance in the multitask setting<br>

Once again, we can apply the weighted loss that we have covered earlier<br>

This time, we not only have a weight associated with just the positive and negative labels, but it's
for the labels associated with a particular class of disease<br>

That is, for `mass`, there will be a different weight to the positive class, than to `pneumonia`,
or to `edema`<br>

<code>L(X, y<sub>mass</sub>) = —w<sub>p, mass</sub> log P(Y = 1|X), if y = 1</code><br>

<code>L(X, y<sub>mass</sub>) = —w<sub>n, mass</sub> log P(Y = 0|X), if y = 0</code><br>

This covers the solution to the second challenge of multitask learning.<br><br>

Let's look at the third challenge which is the **Dataset Size** challenge<br>

For many medical imaging problems, the architecture of choice is the **CNN**,
a.k.a. *ConvNet*<br>

These are designed to process 2D images like X-rays, but variants of these are also
well-suited to **medical signal processing** or **3D medical images** like
**CT scans**, which we'll look at in a future week<br><br>

Several CNN architectures such as **Inception-v3, ResNet-34, DenseNet, ResNeXt and
EfficientNets** have been proposed and are widely popular in image classification<br>

These architectures are composed of various *building blocks*<br>

In medical problems, the standard is to try out multiple models on the desired tasks
and see which ones work best<br><br>

***

# 14. Working with a Small Training Set

The challenge is that all of these architectures are data hungry and benefit from the
`millions of examples with labels` found in image classification datasets<br>

In medical problems, how can we still apply these techniques when we don't have millions of
examples?<br><br>

One solution is to **pre-train the network**<br>

Here the idea is to first have the network look at natural images and learn to
identify objects such as penguins or cats or dogs, then use this network as a
starting point for learning in medical imaging tasks by **copying over (transfering)
the learned features**.<br>

The network can then further be trained (**fine-tuning**) to look at chest X-rays and identify the
presence and absence of diseases<br><br>

The idea of this process is that when we're learning our first task of identifying
cats or dogs, the network will learn general features that will help it's learning on the
medical task<br>

An example of this might be that the features that are useful to identify the edges on a
penguin, are also useful for identifying edges on a lung, which are then helpful to
identify certain diseases<br>

Then when we transfer these features to our new network, the network can learn the new task
of chest X-ray interpretation with a better starting point<br>

The first step is called `pre-training` and the second step is called `fine tuning`<br><br>

It is generally understood that the `early layers of the network` capture low-level image features
(**general features**) that are broadly generalizable, while the `later layers` capture details that are
more **high-level** or more specific to a task<br>

For instance, the early layer might learn about the edges of an object and this might be useful for
chest X-ray interpretation later; but the later layers might learn how to identify the head of a
penguin and may not be useful for chest X-ray interpretation<br>

So when we fine-tune the network on chest X-rays, instead of fine-tuning all features we've
transferred  we can **freeze the features learned by the shallow layers** and just fine-tune the
deeper layers.<br>

In practice, two of the most common design choices are —<br>

1. To fine-tune all of the layers

2. Only fine-tune the later or last layers and not fine-tune the earlier layers<br><br>

This approach of pre-training and fine-tuning is also called **transfer learning**
and is an effective way to tackle the small dataset size challenge<br><br>

***

# 15. Generating More Samples

Let's talk about a second solution to the Dataset Size challenge<br>

The idea is to `trick the network into thinking that we have more training
examples at our disposal than we actually do`<br>

```
~10k or ~100k Images with Labels ➜ ~100k or ~1M Images with Labels
```
<br>

Right before we pass an X-ray image into the network, we can apply a
transformation to it<br>
We have several options here<br>

We can **rotate** it and pass it in, or we can **translate it sideways** and pass it in,
or we can **zoom in**, or we can change the **brightness or contrast**, or apply a
combination of these transformations<br>

This method is called **data augmentation**<br><br>

In practice, there are 2 questions that drive the choice of transformations we pick —<br>

1. Whether we believe the transformation reflects variations that will help the model
   generalize the test set and therefore the real world scenarios<br>

   For instance, we might believe that we're likely to see variations in contrast in
   natural X-rays, so we might have a transformation that changes the contrast of an image<br>

   `Do Augmentations Reflect Variations in Real World?`<br>

2. A second design choice is to verify that our transformation keeps the label the same<br>

   For instance, if we're *laterally inverting* a patient's X-ray (flipping the left over to the right,
   and vice-versa), then their heart would appear on the left-hand side of the image, and the right
   of the body <br>

   However, the label of `Normal` would no longer hold because this is actually a rare heart condition
   called **Dextrocardia** in which your heart points to the right side of your chest instead of to the
   left side<br>

   So this is not a transformation that preserves the label (`Normal` becomes `Normal or Dextrocardia?`)<br>

   The key here is we want the network to learn to recognize images with these transformations that still have
   the same label, not a different one<br>

   `Do Augmentations Keep the Label the Same?`<br><br>

Beyond X-rays, there are other useful data augmentation procedures for other tasks<br>

**Rotation** and **flipping** are useful for algorithms trained to detect skin cancer, for instance<br>

In histopathology, one large source of real-world variation is different shades of pink and purple
seen in these microscopic images [in slide]<br>

**Color noise** is often added and all this does is have slightly different shades of these pink and
purple colors to help the network generalize<br>

In addition, **rotation and cropping** are also useful data augmentation procedures for histopathology
images<br><br>

As a recap, we've looked at the **Weighted Loss** and the **Resampling** methods to tackle the `Class Imbalance challenge`<br>

We've looked at the **Multi-Label Loss** to allow the network to identify multiple diseases in a chest
X-ray (`Multi-Task Challenge`)<br>

We've also covered **Transfer Learning + Data Augmentation** procedures as ways to tackle the
`Small Dataset Challenge`<br><br>

***

# 16. Model Testing

Now that you've seen how you would go about training a model for medical diagnosis, let's talk about
how you would go about testing such a model<br>

You will learn about the proper use of training, validation, and test sets, &
about the need for strong *ground truth* in order to evaluate your models<br><br>

When we apply ML to a dataset, we usually split it into training and test sets<br>

Our **training set** is used for the `development and selection of models` and
our **test set** for the final `reporting of results`<br>

In reality, the training dataset is further split into a training set and a validation
set where the training set is used to learn a model and the **validation set** is used for
hyperparameter tuning and giving an estimate of the model performance on the test set
(`tuning and selection of models`)<br><br>

Sometimes, the split into a training and validation set is done multiple times in a method
called **cross-validation** to `reduce variability in our estimate of the model performance`<br>

These sets also go by different names sometimes, like validation set can be called *tuning or dev set*,
training set can be called the *development set*, and the test set can go by *holdout* or even more
confusing, the *validation set*<br>

We will stick to the terms `training, validation, and test set` for our purposes<br><br>

We will cover 3 challenges with building these sets in the context of medicine<br>

1. The first challenge relates to how we make these test sets independent (**Patient Overlap**)

2. The second relates to how we sample them (**Set Sampling**)

3. The third relates to how we set the ground truth (**Ground Truth**)
<br>

Let's cover the problem of **Patient Overlap** first<br>

Let's say a patient comes in twice for an X-ray, once in June and once in November<br>

Both times, they're wearing a necklace when they have their X-ray taken<br>

One of their X-rays is sampled as part of the training set and the other as part of test<br>

We train our DL model and find that it correctly predicts `normal` for the X-ray in the test set<br>

The problem is that it's possible that the model actually memorized to output `normal` when it saw
the patient with a necklace on<br><br>

This is not hypothetical, `DL models can unintentionally memorize training data`, and the model could
memorize rare or unique training data aspects of the patient, such as the necklace, which could help
it get the right answer when testing on the same patient<br>

This would lead to an **overly-optimistic test set performance**, where `we would think that our model is
better than it actually is`<br><br>

***

# 17. Splitting data by patient

To tackle this problem in our dataset, we can make sure that a patient's X-rays only occur in one of the sets<br>

Now if the model memorizes the necklace on the patient, it does not help it achieve a higher performance on the test
set because it doesn't see the same patient<br>

Let's see this in action over all patients<br><br>

[ ***Split By Image*** ]<br>

**Training Set**<br>

Image | Patient ID | Label(Mass)
----- | ---------- | -----------
xray1.jpg | 20 | 1
xray4.jpg | 17 | 0
xray7.jpg | 32 | 0

**Validation Set**<br>

Image | Patient ID | Label(Mass)
----- | ---------- | -----------
xray2.jpg | 20 | 1
xray5.jpg | 11 | 0
xray9.jpg | 32 | 0

**Test Set**<br>

Image | Patient ID | Label(Mass)
----- | ---------- | -----------
xray0.jpg | 20 | 1
xray3.jpg | 38 | 0
xray8.jpg | 32 | 0

<br>

When we **split a dataset in the traditional way**, images are randomly assigned to one of the sets<br>

Notice that this way we get X-rays that belong to the same patient in different sets<br>

For instance, `xray1` belongs to patient 20 and is part of training<br>
`xray2` also belongs to patient 20 and is part of validation<br>
`xray0` also belongs to patient 20, which is part of test<br>

This is the problem of **patient overlap**<br><br>

[ ***Split By Patient*** ]<br>

**Training Set**<br>

Image | Patient ID | Label(Mass)
----- | ---------- | -----------
xray1.jpg | 20 | 1
xray2.jpg | 20 | 1
xray0.jpg | 20 | 1

**Validation Set**<br>

Image | Patient ID | Label(Mass)
----- | ---------- | -----------
xray7.jpg | 32 | 1
xray8.jpg | 32 | 0
xray9.jpg | 32 | 0

**Test Set**<br>

Image | Patient ID | Label(Mass)
----- | ---------- | -----------
xray4.jpg | 17 | 0
xray3.jpg | 38 | 0
xray5.jpg | 11 | 0

<br>

Instead, when we **split a dataset by patient**, all of the X-rays that belong to the
same patient are in the same set<br>

For instance, `xray1`, `xray2`, and `xray0`, which all belong to patient 20 are all part of training<br>

`xray7`, `xray8`, and `xray9` which are all part of patient 32 are all in the validation set and we
don't see 20 and 32 here in the test set<br>

This way, we can make sure there's no patient overlap between the sets<br>

This covers our solution to the first challenge<br><br>

***

# 18. Sampling

Let's cover the second challenge of Set Sampling.<br><br>

[ ***Dataset*** ]<br>

```
~10k or ~100k Patients with Labels
```

Image | Patient ID | Label (Mass)
----- | ---------- | ------------
xray10.jpg | 15 | 0
xray23.jpg | 24 | 0
xray31.jpg | 20 | 0
xray41.jpg | 56 | 1
... | ... | ...

<br>

[ ***Sampled Set*** ]<br>

Image | Patient ID | Label (Mass)
----- | ---------- | ------------
xray4.jpg | 17 | 0
xray3.jpg | 38 | 0
xray5.jpg | 11 | 0
... | ... | ...

<br>

Let's say we sampled a test set from the dataset<br>

Sometimes, the size of the test set is a fraction of the full dataset, like 10%<br>

Other times, in human comparison studies, because the test set needs to be annotated by
human readers, the bottleneck for the test set size is *how many examples can readers be
expected to read?*<br><br>

<code>
<b>Sizes in Studies</b><br>
120 CT scans, 400-500 X-Rays, 130 Whole Slides
</code>
<br><br>

Typically, test sets contain at least hundreds of examples in medical AI studies<br>

The **challenge with sampling a test set** is that when we're randomly sampling a test set of
hundreds of examples from the dataset, `we might not sample any patients that actually have
a disease.`<br>

Here, we might not sample any examples where the label for `mass` is 1<br>

Thus, we would have no way to actually test the performance of the model on these positive cases<br>

This is especially a problem with medical data where we might already have a small dataset and not that
many examples of each disease<br><br>

One way that this is tackled when creating test sets is to `sample a test set such that we have at least
X% of examples of our minority class`<br>

Here, the minority class is simply the class for which we have few examples, like examples of where `mass` is present.<br>

`One common choice of X is 50%`, so for sampling a dataset of 100 examples, we would have 50 examples of `mass`
and 50 of `normal`<br>

This ensures that the study will have sufficient numbers to get a good estimate of the performance of the model both on
non-disease and on disease examples<br><br>

Once we sample the test set, typically the validation set is sampled next before the training set<br>

Because we want our validation set to reflect the distribution in the test set, typically
the same sampling strategy is used<br>

```
Sample Test Set, then Validation, then Training

Sample to have same distribution of classes as the test set
```
<br>

We might decide to have once again 100 examples in the validation set
of which 50 are `mass` and 50 are `normal`<br>

Finally, the remaining patients can be included in the training set<br>

```
Remaining Patients in Training Set
```

Because the test and validation sets have been artificially sampled to have
a large fraction of mass examples, the training set will have a much smaller
fraction of mass examples<br>

You have seen that we can still train a model in the presence of imbalanced data<br>

So this covers our second challenge of set sampling<br><br>

***

# 19. Ground Truth and Consensus Voting

One major question in testing a model is how we determine the correct label for an
example<br>

The right label is more commonly called the **ground truth** in the context of
ML or the **reference standard** in the context of medicine<br>

On a chest X-ray, differentiating between some diseases might be complex<br>

We might have one expert say this is pneumonia, another experts say it's an another
disease<br>

This is called **inter-observer disagreement**, and is common in medical settings<br>

The **challenge** here is `how we can set the ground truth required for the evaluation of
algorithms in the presence of inter-observer disagreement`<br><br>

So how do we determine the ground truth in the presence of inter-observer disagreement?<br>

We can use the **Consensus Voting** method<br>

The idea behind consensus voting is to `use a group of human experts to determine the
ground truth`<br>

In one setting, we can have 3 radiologists look at a chest X-ray and each determine
whether there is pneumonia present or not<br>

If two out of the three say *yes*, then we would say the answer is *yes*<br>

In general, the answer will be the `majority vote of the three radiologists`<br>

Alternatively, we can have the three radiologists get into a room and discuss
their interpretation until they reach a single decision, which can then be used
as the ground truth<br><br>

***

# 20. Additional Medical Testing

The second method is to use a more definitive test which provides additional information
to set the ground truth<br>

For example, to determine whether a patient has a mass using a chest X-ray, a more definitive
test that can be performed is a **CT Scan**<br>

The CT scan `shows the 3D structure of the potential abnormality`, thus giving the radiologist
more information<br>

If a mass is confirmed on the CT, we can then assign that ground truth to the chest X-ray<br><br>

In the dermatology study we saw earlier, the ground truth for the test set was determined by
a **skin lesion biopsy**<br>

This is a medical procedure in which `a sample of the skin with the suspected cancer is removed
and tested in a lab`<br>

The results of this test are then used to set the ground truth for the photo image<br><br>

We've now looked at 2 methods to set the ground truth or reference standard — **consensus
voting and having a more definitive test**<br>

With the second method, the difficulty is that we might not have these additional tests available<br>

Not everyone who gets a chest X-ray gets a CT scan, and not everyone who has a potentially suspicious
skin lesion gets a biopsy<br>

Thus having a reliable ground truth with an existing data set often has to `use the first method
of getting a consensus ground truth on the existing data`, which is the strategy that many medical
AI studies use<br><br>

Thus we have covered our 3 challenges with algorithm testing and some solutions in the context of medicine<br>

1. **Patient Overlap Challenge** — `Split by patient` in the creation of training, validation, and test sets

2. **Set Sampling Challenge** — `Sample a minimum percent of a minority class` in the test set, and

3. **Ground Truth Challenge** — Use `consensus voting or a more definitive test` to set the ground truth
<br>

Congratulations on completing the first week!<br>

This week you've learned about examples of DL models for tasks across several fields of medicine, &
have learned about all the techniques needed to both train and test your own high performance model for
chest X-ray interpretation<br>

You'll get to implement these ideas in the assignment and have your own chest X-ray interpretation model<br>

In the next week, you'll learn about ideas and methods to evaluate the models that you've built<br>

See you then<br><br>

***
