☀ 1. [Sensitivity, Specificity, and Evaluation Metrics](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#1-sensitivity-specificity-and-evaluation-metrics)<br>

☀ 2. [Accuracy in terms of conditional probability](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#2-accuracy-in-terms-of-conditional-probability)<br>

☀ 3. [Sensitivity, Specificity and Prevalence](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#3-sensitivity-specificity-and-prevalence)<br>

☀ 4. [PPV, NPV](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#4-ppv-npv)<br>

☀ 5. [Confusion matrix](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#5-confusion-matrix)<br>

☀ 6. [ROC curve and Threshold](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#6-roc-curve-and-threshold)<br>

☀ 7. [Varying the threshold](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#7-varying-the-threshold)<br>

☀ 8. [Sampling from the Total Population](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#8-sampling-from-the-total-population)<br>

☀ 9. [Confidence intervals](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#9-confidence-intervals)<br>

☀ 10. [95% Confidence interval](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%201%20%E2%80%94%20AI%20for%20Medical%20Diagnosis/Week%202%20%E2%80%94%20Evaluating%20Models/Week-2%20Notes.md#10-95-confidence-interval)<br><br>

# 1. Sensitivity, Specificity, and Evaluation Metrics

This week, we'll dive further into the evaluation of DL models for medicine.<br>

In medicine, because the decisions are high impact, we care about
understanding exactly when a model works on a patient and when it does not.<br>

You'll learn about metrics including **sensitivity, specificity, predictive values and the ROC curve** which are key ingredients in
evaluating models in medical settings.<br><br>

In this lesson, we'll talk about the shortcomings of **accuracy** as a
metric<br>

From accuracy, we'll look at how we can derive sensitivity and specificity, 2 core concepts in medical evaluation<br>

In the second part of this lesson, we'll talk about **predictive values**,
which can `help medical professionals with clinical decision-making`<br><br>

## Evaluation Metrics

To answer the question of how good a model is, we'll start by looking at
accuracy<br>

When computing the **accuracy** on a test set, we look at the `proportion of the total examples that the model correctly classified`<br>

```
Accuracy = (Examples correctly classified) / (Total number of examples)
```

Let's work with an example so we can illustrate the computation of accuracy<br><br>

```
Ground Truth = [Normal, Normal, Normal, Normal, Normal,
                Disease, Normal, Disease, Normal, Normal]
```

Here we have a test set of 10 examples — 8 of them have the ground truth of
`Normal` and 2 have the ground truth of `Disease`<br>

Let's say we have a model which outputs negative for all 10 patients<br>

*'Negative'* here means the model is outputting the prediction of `Normal`<br>

This is of course not a useful model, but notice that it's getting
all the `Normal` examples right, so it's getting 8 / 10 examples
right and thus has an **accuracy of 0.8**<br><br>

```
Model 1 = [—, —, —, —, —, —, —, —, —, —]

Model 2 = [—, +, —, —, —, +, —, +, +, —]
```

Compare that to Model 2.<br>
Model 2 correctly predicts *'Positive'* on the 2 `Disease` examples here and also calls 2 of the `Normal` examples *'Positive'*<br>

Now we can compute the accuracy of the model 2 and if we go through this computation, we'll find once again that we get 8 / 10 examples right with model 2 to get an **accuracy of 0.8**<br>

So we have 2 models with an accuracy of 0.8<br>

Although we haven't formalized this, we get a sense that **model 2** is perhaps `doing something more useful` than model 1 because it is at least `attempting to distinguish between healthy and diseased patients.`<br><br>

***
<br>

# 2. Accuracy in terms of conditional probability

Let's understand the accuracy metric a little more<br>

We'll see how we can use the accuracy to derive other useful evaluation metrics such as sensitivity and specificity<br><br>

We'll start by interpreting **accuracy** as a `probability of being correct`<br>

We can break down this probability of being correct as the sum of 2 probabilities<br>

The probability that the model is correct and a patient has the `Disease` + the probability that the model is correct and the patient is `Normal`<br>

```
Accuracy = P(correct)

Accuracy = P(correct ⋂ disease) + P(correct ⋂ normal)
```
<br>

The **law of conditional probability** allows us to further expand this out<br>

As a reminder, the law says that `the probability of A and B is the probability of A given B x the probability of B`<br>

We can apply this to the first and second terms & expand them out<br>

```
Using P(A ⋂ B) = P(A | B) x P(B)

Accuracy = P(correct | disease) P(disease) + P(correct | normal) P(normal)
```
<br>

What this allows us to do is interpret the terms `P(correct | disease)` and `P(correct | normal)`<br>

The probability of being correct given the patient has the `Disease` means that we predicted *'Positive'*<br>

So we can replace the first term with the probability that we predict
*'Positive'* given a patient has `Disease`<br>

Similarly, the probability of being correct when the patient is `Normal`
would mean that we predicted *'Negative'*, so we can replace the second term by the probability of predicting *'Negative'* given a patient is `Normal`<br>

```
Accuracy = P(+ | disease) P(disease) + P(— | normal) P(normal)
```
<br>

***
<br>

# 3. Sensitivity, Specificity, and Prevalence

Out of this expression pop out 2 very important quantities that are the bread and butter of evaluation medicine — Sensitivity and Specificity<br>

These terms are also known by *True Positive Rate* and *True Negative Rate* which refer to the same 2 quantities<br>

```
P(+ | disease) = Sensitivity (True Positive Rate)

P(— | normal) = Specificity (True Negative Rate)
```
<br>

**Sensitivity** is the <code>probability that the model classifies a patient as having the Disease given that they have the Disease</code><br>

Sensitivity = *If a patient has the disease, what is the probability that the model predicts Positive?*<br><br>


**Specificity** is the <code>probability that the model classifies a patient as being Normal given that they are Normal</code><br>

Specificity = *If a patient is normal, what is the probability that the model predicts negative?*<br><br>

Thus, we've broken our global measure of Accuracy down into the useful quantities of Sensitivity and Specificity<br>

The other piece of useful terminology is for the remaining terms<br>

The `probability of a patient having Disease in a population` is called **Prevalence**<br>

The probability of being `Normal` is simply  (1 — probability of having disease) or (1 — Prevalence)<br>

We can thus write Accuracy in terms of Sensitivity, Specificity, and Prevalence<br>

```
Accuracy = P(correct)

Accuracy = Sensitivity x P(disease) + Specificity x P(normal)

Accuracy = Sensitivity x Prevalence + Specificity x (1 - Prevalence)
```
<br>

Why is this useful?<br>
This equation allows us to see Accuracy as a `weighted average of Sensitivity and Specificity`<br>

The weight associated with the Sensitivity is the Prevalence, and the weight associated with the Specificity is (1 — Prevalence)<br>

This equation also allows us to find out any of these quantities given the other 3 quantities<br>

Let's see this using an example<br><br>

```
Ground Truth = [Normal, Normal, Disease, Normal, Normal,
                Disease, Normal, Disease, Normal, Normal]

Model = [—, —, +, —, —, —, —, +, +, —]
```

First, we compute Sensitivity<br>
Sensitivity can be computed as the fraction of `Disease` examples that are also *'Positive'*<br>

```
Sensitivity

#(+ and Disease) / #(Disease) = 2 / 3 = 0.67
```
<br>

Then we compute Specificity<br>
Specificity can be computed as the fraction of `Normal` examples that are also *'Negative'*<br>

```
Specificity

#(— and Normal) / #(Normal) = 6 / 7 = 0.86
```

So that's the computation of Sensitivity and Specificity<br><br>

Now let's look at the prevalence of `Disease` in this set<br>

This is simply the fraction of `Disease` examples computed as the number of `Disease` examples over the total number of examples<br>

```
Prevalence

P(disease) = #(Disease) / #(Total) = 3 / 10 = 0.3
```
<br>

Now using the relationship between Sensitivity, Specificity, and Prevalence, we can also get to the Accuracy<br>

```
Accuracy

Sensitivity x Prevalence + Specificity x (1 — Prevalence)
= 0.67 x 0.3 + 0.86 x 0.7
= 0.8
```

We can confirm with a simple calculation of the fraction that we get correct that the Accuracy is indeed 0.8<br><br>

Thus, we have covered 2 metrics we can use to evaluate diagnostic AI models including Sensitivity and Specificity & we've seen how they're
related now to Accuracy<br><br>

***
<br>

# 4. PPV, NPV

**Sensitivity** tells us given we know a patient has a `Disease`, what is
the probability that the model predicts *'Positive'*?<br>

```
Sensitivity

P(+ | Disease)
```
<br>

In the clinic, the doctor using an AI model may be interested in a
different question and that is, given the model predicts *'Positive'* on
a patient, what is the probability that they actually have the
`Disease`?<br>

```
If a model prediction is Positive, what is the probability that a patient has the Disease?

P(Disease | +)
````

This is called the **Positive Predictive Value or PPV** of the model<br><br>

Similarly, while Specificity asks what is the probability the model predicts *'Negative'* give a patient is `Normal`..<br>

```
Specificity

P(— | Normal)
```
<br>

..the Doctor may be interested in knowing the probability that a patient is `Normal` given the model prediction is *'Negative'*<br>

```
If a model prediction is Negative, what is the probability that a patient
is Normal?

P(Normal | —)
```

This is called the **Negative Predictive Value or NPV** of the model<br><br>

Let's compute the PPV and NPV on an example<br>

Once again, we have 10 examples on which a model has made its
predictions<br>

```
Ground Truth = [Normal, Disease, Normal, Normal, Normal,
                Disease, Normal, Disease, Normal, Normal]

Model = [—, +, +, —, —, —, —, +, +, —]
```
<br>

First, let's compute PPV<br>

PPV can be computed as the fraction of *'Positive'* examples that are also `Disease`<br>

```
PPV

P(Disease | +) = #(+ and Disease) / #(+) = 2 / 4 = 0.5
```
<br>

Let's now look at NPV<br>

NPV can be computed as the fraction of *'Negative'* examples that are also
`Normal`s<br>

```
NPV

P(— and Normal) = #(— and Normal) / #(—) = 5 / 6 = 0.83
```
<br>

Now that we've seen PPV and NPV, in addition to Sensitivity and Specificity, let's look at how they relate to each other<br>

```
PPV             ===    P(Disease | +)

NPV             ===    P(Normal | —)

Sensitivity     ===    P(+ | Disease)

Specificity     ===    P(— | Normal)
```

***
<br>

# 5. Confusion Matrix

To see their relation, we'll use the confusion matrix.<br>

The **Confusion Matrix** can be `used to look at the performance of a classifier in the form of a table`<br>

The rows correspond to the Ground Truth and the columns to the model
predictions or the model outputs<br>

The cells in the table correspond to the number of elements
corresponding to each Ground Truth model prediction combination<br><br>

```
Ground Truth = [Normal, Disease, Normal, Normal, Normal,
                Disease, Normal, Disease, Normal, Normal]

Model = [—, +, +, —, —, —, —, +, +, —]
```
<br>

[ **Model Output** ]<br>

Ground Truth | + | —
----- | - | -
Disease | 2 | 1
Normal | 2 | 5

For example, the 2 in the first row represents the counts where we
have a Ground Truth of `Disease`, and the model output of *'Positive'*
at the same time<br>

The sum of the 4 cells should be the total number of examples, which
is 10<br><br>

These four counts are more generally known as **True Positives, False
Negatives, False Positives, and True Negatives** *[row-first
order]*<br>

Ground Truth | + | —
----- | - | -
Disease | True Positive (TP) | False Negative (FN)
Normal | False Positive (FP) | True Negative (TN)

<br>

We've seen the formulas for the computations of each of these metrics
in terms of counts<br>

Note that they directly correspond to cells in the Confusion Matrix<br>

For instance, **PPV** deals with the count of the top-left cell,
divided by summing the entire first column that gives the total
number of examples where a `Positive` prediction was made<br>

Another metric, **Specificity** is looking at the number of
*'Negative'* and `Normal` (bottom-right cell) divided by summing the
second row (all of the examples where the Ground Truth is
`Normal`)<br>

So we can represent all of these metrics in terms of the four terms —
TP, FN, FP, TN<br><br>

Here's the transformation of the formulas to use the terms in the
Confusion Matrix<br>

```
Sensitivity = TP / (TP + FN)

Specificity = TN / (FP + TN)

PPV = TP / (TP + FP)

NPV = TN / (FN + TN)
```

Thus, the Confusion Matrix can be used to derive all of these 4
metrics that we've looked at to evaluate the performance of a
model.<br><br>

***
<br>

# 6. ROC curve and Threshold

In this lesson, we'll look at one of the most useful tools to
evaluate medical models, the **ROC curve**<br>

We'll see how the ROC curve `allows us to visually plot the
Sensitivity of a model against the Specificity of the model at
different decision thresholds`<br><br>

A chest X-ray classification model outputs a probability of `Disease`
given an X-ray<br>

This output can be transformed into a diagnosis using a **Threshold
or Operating Point**<br>

When the probability is above a Threshold, then we interpret this as
*'Positive'* or saying the patient has the `Disease`<br>

When the probability is below the Threshold, then we interpret this
as *'Negative'* or saying the patient is `Normal`<br><br>

For example, if our score is 0.7 and our Threshold is 0.5, then we
would classify this example as *'Positive'*<br>

But if our score was 0.2 and our Threshold was 0.5, we would classify
this example as *'Negative'*<br><br>

Our choice of Threshold affects the metrics we have looked at thus
far<br>

For example, **if we had a Threshold of 0**, then we would classify
everything as *'Positive'* and so our `Sensitivity would be 1`, while
our `Specificity would be 0`<br>

Similarly, **if we had chosen a Threshold of 1**, we would classify
everything as *'Negative'* so our `Specificity would be 1`, while our
`Sensitivity would be 0`<br><br>

Let's dive further into how our choice of Threshold, a.k.a. the
*Operating Point* affects these quantities<br><br>

***
<br>

# 7. Varying the Threshold

Let's say we have a test set of 15 chest X-rays, which we ran through
our model to get an output probability or a *Score* for each of them<br>

X-Ray | Output Probability (Score)
----- | --------------------------
1 | 0.30
2 | 0.42
3 | 0.78
... | ...
15 | 0.98

We can plot these 15 outputs' scores on a number line between 0 and 1<br>

Now some of these X-rays will have a Ground Truth of `Disease` and
some will be `Normal`, so let's color them accordingly<br>

Here `Disease` is red and `Normal` is blue<br>

```
—, —, —, —, —, +, —, +, —, +, +, +, —, +, +
```
<br>

We can pick a Threshold (*t*) that sets everything on the right of
the Threshold as we classify *'Positive'* and everything on the left
of the Threshold, we classify as *'Negative'*<br>

```
—, —, —, —, —, +, — ||| +, —, +, +, +, —, +, +
```

Now notice we can compute the Sensitivity and Specificity of the model<br><br>

The denominator for **Sensitivity** is the total number of `Disease`
examples, which we can count as the total number of red === 7<br>

The numerator is how many of those are *'Positive'* i.e. on the right
side of the Threshold === 6<br>

So our Sensitivity is 6 / 7 === `0.85`<br><br>

Similarly, the denominator for **Specificity** is the total number of
`Normal` examples, which is the total number of blue === 8<br>

The numerator is how many of those are *'Negative'* i.e. on the left
side of the Threshold == 6<br>

So our Specificity is 6 / 8 = `0.75`<br><br>

Let's say we now change the Threshold such that it was now higher<br>

```
—, —, —, —, —, +, — , +, — ||| +, +, +, —, +, +
```

We now expect we classify fewer examples as *'Positive'* and more
examples as *'Negative'*<br>

We can now recompute the Sensitivity and Specificity<br>

```
Sensitivity = P(+ | Disease) = 5 / 7 = 0.71

Specificity = P(— | Normal) = 7 / 8 = 0.88
```
<br>

Note that the **Sensitivity has gone down**, our numerator has fallen,
and the **Specificity has gone up**, our numerator has increased
because we are now correctly classifying more `Normal` patients and
incorrectly classifying more `Disease` patients<br><br>

We can take this to the extreme and set the Threshold to be at 1<br>

```
—, —, —, —, —, +, — , +, —, +, +, +, —, +, + |||
```

In this case, Sensitivity is going to be 0 since no examples are
classified *'Positive'* and Specificity is going to be 1, since all
the examples are classified as *'Negative'*<br>

```
Sensitivity = P(+ | Disease) = 0 / 7 = 0

Specificity = P(— | Normal) = 8 / 8 = 1
```
<br>

***
<br>

# 8. Sampling from the Total Population

In this lesson, we'll look at another very important aspect of
evaluating medical models and that is **reporting the variability in
our estimate**<br>

We will look at how **confidence intervals** can be used to show this
variability<br><br>

Let's say there were 50,000 patients in a hospital and we wanted to
find out the accuracy of our chest X-ray model on everyone who gets
a chest X-ray in the hospital<br>

If we were able to run the model and get the Ground Truth for all
patients, we'd be able to get the performance of the model on the
whole population<br>

For example, say we're looking at accuracy (this could be another
metric)<br>

We find that the accuracy of the model around all 50,000 patients
is `0.78`<br>

This is called the **Population Accuracy** (*p*)<br><br>

In reality, we don't want to test the model on the whole population
because it's simply infeasible to do so<br>

Therefore, the Population Accuracy, *p* is unknown<br>

The question is, `can we get a sense of how well the model will
perform on this population by using a small sample of patients?`<br>

Let's say we sample 100 patients (*n = 100*) from the hospital<br>

Now we find that the model gets an accuracy (*p̂*) of 0.8 on the set<br>

<code>Can we say anything about the range in which the Population
Accuracy, *p* will lie?</code><br><br>

***
<br>

# 9. Confidence Intervals

**Confidence Intervals** allow us to say that using our sample, we're
<code>95% confident that the Population Accuracy, *p*, is in the interval [0.72, 0.88]</code><br>

0.72 is called the **Lower Bound** and 0.88 the **Upper Bound** of this
interval<br>

The calculation of these Confidence Intervals is beyond the scope of this
course, but it's important to understand their interpretation<br><br>

When we report the accuracy of a model on the sample, we report it with the
**mean and the Confidence Intervals**<br>

```
0.80 (95% CI 0.72, 0.88)
```
<br>

✔️ The **95% Confidence Intervals (CIs)** here <code>allow us to say that with
95% confidence, *p* is in the interval [0.72, 0.88]</code><br>

What we haven't seen is what it means to be 95% confident<br>

❌ 95% confidence does not say there is a 95% probability that *p* lies within
the interval [0.72, 0.88]<br>

❌ It also does not say that 95% of the sample accuracies lie within this
interval [0.72, 0.88]<br><br>

The interpretation of 95% confidence is a little more nuanced and requires us
to think about making repeated samples<br>

Let's dive into this<br><br>

Let's say we were able to repeatedly sample in 100 patients from the population
several times<br>

Each time we get a different sample and hence a different sample accuracy<br>

We can also compute the Confidence Intervals associated with each sample<br>

```
(p) is unknown

0.79 (95% CI 0.71, 0.87)
0.75 (95% CI 0.67, 0.83)
0.80 (95% CI 0.72, 0.88)
```
<br>

We can look at these samples on a plot<br>

For each of these samples, we can plot the `sample accuracies` (represented
by a circle [in the slide]) & `the lower and upper bounds` of the Confidence
Intervals of the samples<br>

On this plot, we also have the `true Population Accuracy` plotted as the dotted
line (*p = 0.78*)<br>
This is unobserved<br>

Note that most of these samples contain the Population Accuracy, the vertical
line, here 6 out of the 7 contain it and 1 misses it<br><br>

In fact, when we have 95% Confidence Intervals, 95% of the samples will
contain the Population Accuracy<br>

95% is what's called our **Confidence Level**<br>

✔️ Thus, the **interpretation of 95% confidence** is that `in repeated
sampling, this method produces intervals that include the Population Accuracy
in about 95% of samples`<br><br>

***
<br>

# 10. 95% Confidence Interval

In practice, we don't compute the Confidence Intervals for many samples<br>

We only compute our model performance on one sample<br>

For our sample, the computed Confidence Interval may or may not contain
*p*<br>
However, we can be 95% confident that it does<br>

```
(p) Unknown

p̂ = 0.80, n = 100
0.80 (95% CI 0.72, 0.88)
Known
```
<br>

One of the factors that affects the width of the Confidence Intervals, which
is given by how close these numbers are, is the **sample size**<br>

Let's say we drew another sample from the population, but this time with 500
patients — 5x as large as our previous sample<br>

We can expect that we'll have a `better estimate of the Population Accuracy`,
*p*, using the larger sample<br>

We can see that even though the model gets an accuracy of 0.8 on both samples,
notice that the `Confidence Intervals are tighter for the larger sample` and
wider for the smaller sample<br>

Thus, a larger sample is giving us a better estimate of this Population
Accuracy, because these numbers are closer together<br>

```
p̂ = 0.80, n = 100
0.80 (95% CI 0.72, 0.88)

p̂ = 0.80, n = 500
0.80 (95% CI 0.76, 0.84)
```
<br>

To summarize, Confidence Intervals are useful because even when we cannot run
the model on a whole population, we can `at least use a test result on a
sample to express the range in which we're pretty sure our Population
Accuracy lies.`<br><br>

Congratulations on completing this week on model evaluation!<br>

As you've seen, we need a lot more than accuracy to be able to properly
evaluate models in medicine because we care about understanding exactly
when a model works on patients and when it does not<br>

In this week's assignment, you'll be able to apply these ideas to evaluate
your chest X-ray model more comprehensively<br><br>

Next week, we'll be jumping from medical image classification to **medical
image segmentation** where you'll be building a brain tumor segmentation model
from MRI data<br>

See you then<br><br>

***
