☀ 1. [Course-2 Introduction with Andrew and Pranav](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#1-course-2-introduction-with-andrew-and-pranav)<br>

☀ 2. [Prerequisites and Learning Outcomes](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#2-prerequisites-and-learning-outcomes)<br>

☀ 3. [Medical Prognosis](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#3-medical-prognosis)<br>

☀ 4. [Examples of Prognostic Tasks](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#4-examples-of-prognostic-tasks)<br>

☀ 5. [Atrial Fibrillation](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#5-atrial-fibrillation)<br>

☀ 6. [Liver Disease Mortality](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#6-liver-disease-mortality)<br>

☀ 7. [Risk of Heart Disease](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#7-risk-of-heart-disease)<br>

☀ 8. [Risk Score Computation](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#8-risk-score-computation)<br>

☀ 9. [Evaluating Prognostic Models](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#9-evaluating-prognostic-models)<br>

☀ 10. [Concordant Pairs, Risk Ties, Permissible Pairs](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#10-concordant-pairs-risk-ties-permissible-pairs)<br>

☀ 11. [C-Index](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/Week-1%20Notes.md#11-c-index)<br><br>

# 1. Course-2 Introduction with Andrew and Pranav

Welcome back!<br>
This second course of this Specialization is focused on
Medical Prognosis<br>

**Prognosis** is a `branch of medicine that specializes in
predicting the future health of patients`<br>

For example, given a patient's lab results, can you estimate
the risk of having a heart attack over the next 5 years or
the risk of dying over the next 10 years?<br>

In this course, you also get to practice building **Decision
Trees** and **Random Forests** using structured data<br>

Machine Learning is a powerful tool for prognosis, and can
provide a tremendous boost to this branch of medicine by using
many different types of medical data to make accurate
predictions about a patient's future health<br><br>

In this first week, you'll learn what is Prognosis and we'll
see multiple examples of prognostic tasks including a few
examples where `Prognosis using Risk calculations is part of
routine clinical practice`<br>

In the second week, you will build ML models with Decision
Trees<br>

You will use trees to model non-linear relationships, which
are commonly observed in medical data and apply them to the
prognostic task of <code>predicting **Mortality**, that is the
chance of a patient dying</code><br>

In practice, when we train ML models, one of the `key
challenges` is **how to handle missing data**<br>

You'll learn about a few ways of dealing with missing data in
your ML pipelines<br><br>

In Week-3, you'll learn about **Survival Models**<br>

Say a patient has a particular type of cancer and you'd like
to estimate the probability of their surviving 1 year, 2
years or 5 years, or even longer<br>

This is when you use a Survival Model<br>
It `allows you to model the time to an event`, in this case
the patient's possible death<br>

These models help doctors answer patient questions like,
*How likely am I to survive the next 5 years or the next 10
years?*<br>

Survival Models are also used to model the time from treatment
to recurrence, so questions like *How likely am I to get a
recurrence of this cancer in 1 year or in 2 years?*<br><br>

In Week-4, you will learn about strategies to build and
evaluate Survival Models that allow you to compare the risks
of individual patients<br>

You will learn about 2 such models — the **Cox proportional
hazards model** and **Survival Trees**<br>

Finally, you'll learn about a way to evaluate the prediction
performance of the survival prediction models that you built.<br>

As we collect larger and larger medical datasets, ML will
become an invaluable tool to learn the complex relationships
in medical data, and to help us answer questions like
*why some people survive longer than others*, or *what is the
patient's 10-year risk of heart attack?*<br><br>

Let's dive in<br><br>

***
<br>

# 2. Prerequisites and Learning Outcomes

I want to talk a little bit about the requirements for this
course<br>

This course has been designed with a strong focus on equipping
you with the concepts and the practical tools you will need to
successfully build ML models for medicine<br>

For this course, you won't need any background in deep
learning methods, and you don't need any background in
medicine<br>

There are, however, **3 prerequisites** that I suggest you
feel comfortable with before you take this course —<br>

1. The course `assumes that you know the basics of ML`, for
   e.g. you should know the basics of *supervised learning* &
   *splitting your data* into training, validation, and test
   sets <br>

   You should know what *linear* and *logistic* **Regression
   models** are<br>

   If you've taken Course-1, then you should be good<br><br>

2. You `should be fairly comfortable being able to write code
   in Python`, as you'll be using it to process data, build
   models, and interpret results in the assignments<br><br>

3. The lecture concepts will assume `knowledge of
   probability`, for instance, you should know what a
   **Conditional Probability** is

With these 3 prerequisites, you should be good to go<br><br>

AI has the power to transform prognostic medicine<br>

It's going to take individuals like you to realize that
potential<br>

Coming out of this course, you will be equipped with the
tools to effectively apply your knowledge and skills to make a
difference<br><br>

Without further ado, let's dive in<br><br>

***
<br>

# 3. Medical Prognosis

It's first very important for us to talk about what Prognosis
is and why it's important in medical practice<br>

Let's dive in<br><br>

**Prognosis** is a `medical term that refers to predicting
the risk of a future event`<br>

Here, *event* is a general term that captures a variety of
things that can happen to an individual<br>

Events can include outcomes such as Death, and other adverse
events like a Heart Attack or a Stroke, which might be risks
for patients who have a specific medical condition or for the
general population<br>

```
Predicting risk of a future event

Death, Heart Attack, Stroke, etc.
```
<br>

Making prognosis is a clinically useful task for a variety
of reasons<br>

**First**, Prognosis is `useful for informing patients their
risk of developing an illness`<br>

For e.g., there are blood tests that are used to estimate the
risk of developing breast and ovarian cancer<br>

**Second**, Prognosis is also `useful for informing patients
how long they can expect to survive with a certain illness`<br>

An example of this is ***cancer staging***, which gives an
estimate of the survival time for patients with that
particular cancer<br>

```
Informing Patients

1. Risk of illness
2. Survival with illness
```
<br>

Prognosis is also useful for **Guiding Treatment**<br>

In clinical practice, the prediction of the ***10-year risk of
heart attack*** is `used to determine whether a patient
should get drugs to reduce the risk`<br>

Another example is the ***6-month mortality risk***<br>

This is used for patients with terminal conditions that have
become advanced and incurable & is `used to determine who
should receive end-of-life care`<br>

```
Guiding Treatment

1. Risk of heart attack ==> Who should get drugs

2. 6-month Mortality Risk ==> Who should receive end-of-life
care
```
<br>

Now that we know the definition of Prognosis, let's try to
determine whether the following are prognostic tasks —<br>

1. **Estimating the 10-year cardiovascular risk of an
   individual**<br>

   This is a prognostic task because we're predicting the
   future risk of an event, here that's the cardiovascular
   risk<br><br>

2. **Using a Chest CT scan to diagnose Lung Cancer**<br>

   The keyword here is *diagnosis*, which is `the
   identification of an illness`, not the prediction of a
   future event<br><br>

3. **Recommending lifestyle changes to a patient with
   diabetes**<br>

   This is a treatment recommendation, not predicting the
   future risk of an event<br><br>

4. **Forecasting the risk of lung cancer recurrence after
   therapy**<br>

   Here once again, we're predicting the risk of a future
   event, in this case, the recurrence of cancer, so this is
   a prognostic task<br><br>

***
<br>

# 4. Examples of Prognostic Tasks

In this lesson, we will look at clinical examples where
Prognosis is part of routine practice<br>

I want to give you a sense of what the inputs and outputs for
prognostic tasks look like, and how we might weigh some
inputs more than others <br><br>

We can think of **Prognostic Models** as `a system that takes
in a Profile of a patient as input and outputs a Risk Score
for that patient` <br>

The patient Profile can include —<br>
1. **Clinical History** (includes major illnesses, any
previous procedures, etc.),

2. **Physical Examination findings** (such as vital signs
including temperature and blood pressure), &

3. **Labs and Imaging tests**
(such as complete blood count, CT scan and others)
<br>

Prognostic Models can take one or more of these pieces and
output a Risk Score for the patient<br>

Now, **Risk Scores** might be arbitrary numbers or they might
be probabilities<br>

```
Patient Profile {
  Clinical History,
  Physical Examinations,
  Labs and Imaging
}

is passed via a Prognostic Model to get

Risk Score {
  121.0
  11%
  ...
}
```
<br>

Let's see an example of a Prognostic Model<br>

Profile Feature | Value (Yes = 1, No = 0)
--------------- | ----------------------
Smoker | 1
Age 75 years or older | 1
**Score** | 2

<br>

This prognostic model is going to use 2 patient features to
come up with a risk score for heart disease<br>

Let's say we had a patient who was a smoker over the age of
75, we would give them a score of 1 for being a `Smoker` and
1 for `Being over 75` for a final Risk Score of 2<br><br>

Let's make this prognostic model a little more interesting<br>

Profile Feature | Coefficient | Value (Yes = 1, No = 0) | Coefficient x value
--------------- | ----------- | ----------------------- | -----------------------
Smoker | 1 | 1 | 1
Age 75 years or older | 2 | 1 | 2
**Score** | | | **3**

<br>

Let's say `Being over 75` is twice as risky as being a
`Smoker` and we wanted the model to give twice as much
weight to the age as to being a smoker<br>

Then we can assign a weight or a Coefficient to the `Smoker`
feature of 1 and to the `Being over 75` feature of 2<br>

Now, we could multiply these Coefficients by the Value to get
the contribution of each of the features of 1 for being a
`Smoker` and 2 for `Being over 75`, for a total Risk Score
of 3<br><br>

***
<br>

# 5. Atrial Fibrillation

In the first example, we'll look at a risk calculator that is
used for patients with **Atrial Fibrillation**<br>

Atrial Fibrillation is `a common abnormal heart rhythm that
puts the patient at a risk of` ***Stroke***<br>

Stroke is `when blood flow to an area of brain is cut off`<br>

For patients with AF, we'll look at an example of a model
that predicts the **1-year risk of stroke**<br>

```
For patients with Atrial Fibrillation (AF) ==> What's the 1-year risk of Stroke?
```
<br>

This is called the **CHA<sub>2</sub>DS<sub>2</sub>-VASc**
(*chads vasc*) **score**, which is `an abbreviation of the
features that are used as part of the patient profile`<br>

Abbr. | Feature | Coefficient | Value (1 = Yes, 0 = No) | Coefficient x Value
----- | ------- | ----------- | ----------------------- | -----------------------
C | Congestive heart failure | 1 | |
H | Hypertension | 1 | |
A<sub>2</sub> | Age 75 years or older | 2 | |
D | Diabetes mellitus | 1 | |
S<sub>2</sub> | Stroke, TIA, or TE | 2 | |
V | Vascular disease | 1 | |
A | Age 65 to 74 years | 1 | |
Sc | Sex category (female) | 1 | |
| **Score**

<br>

🧬 We will compute the Chads-Vasc score for a patient who is
a 70-year-old male diagnosed with AF, who has *Hypertension*
(high blood pressure), and *Diabetes*<br>

We will calculate their one-year risk of Stroke<br>

There are a lot of medical terms here and it's all right if
you don't know what they mean, as long as you understand that
they are features that are part of a Patient Profile<br>

Note that the Prognostic Model already tell us the
***Coefficients*** or the `weights associated with each of
the features`<br><br>

Our first task is going to be to enter in all of the patient
values into the table<br>

We know that this patient has Hypertension and Diabetes, and
we can fill that in; we also know their age which
activates the `Age 65 to 74 years` feature and we can
enter in zeros for everything else<br>

We will now multiply each of these values by the Coefficient
associated with that value and enter that in into the
`Coefficient x Value` column<br>

Abbr. | Feature | Coefficient | Value (1 = Yes, 0 = No) | Coefficient x Value
----- | ------- | ----------- | ----------------------- | -----------------------
C | Congestive heart failure | 1 | 0 | 0
H | Hypertension | 1 | 1 | 1
A<sub>2</sub> | Age 75 years or older | 2 | 0 | 0
D | Diabetes mellitus | 1 | 1 | 1
S<sub>2</sub> | Stroke, TIA, or TE | 2 | 0 | 0
V | Vascular disease | 1 | 0 | 0
A | Age 65 to 74 years | 1 | 1 | 1
Sc | Sex category (female) | 1 | 0 | 0
| **Score** | | | **3**

<br>

3 is the Risk Score that this model outputs<br>

And so we have computed by hand, our first Prognostic Score
using a model that is used in clinical practice<br><br>

***
<br>

# 6. Liver Disease Mortality

In our second example, we're going to be looking at
**mortality** or death<br>

We will be looking at a score that is used for patients over
the age of 12 that are on liver transplant waiting lists<br>

This score gives an **estimate of the 3-month mortality** for
patients and is `one of the factors that determines how
quickly the patient might get a liver transplant`<br>

```
For patients ≥ 12 on liver transplant waiting lists ==> Provides estimate of 3-month mortality for patients

One factor that determines how quickly a patient might get a
liver transplant
```
<br>

This is the **Model for end-stage liver disease** which
produces what is called the **(MELD) Score**<br>

We will once again compute the MELD score for an example
patient by hand<br>

🧬 Here, we have a 50-year-old woman who has lab studies
showing a few lab results, which we will need in the model<br>

```
A 50-year-old woman has lab results —

* Creatinine = 1.0 mg/dL
* Bilirubin total = 2.0 mg/dL
* INR = 1.1
```
<br>

There are a couple of new features about this model that make
this more interesting<br>

Feature | Coeff. | Value | Coeff. x Value
------- | ------ | ----- | --------------
Ln Creatinine (mg/dL) | 0.957 | |
Ln Bilirubin (mg/dL) | 0.378 | |
Ln INR | 1.120 | |
Intercept | 0.643 | 1 | 0.643
**Score (multiplied by 10)** | | |

<br>

**First**, notice that the first feature is not just a lab
value but is the ***natural log*** of that lab value<br>

Similarly for the other 2 lab values, we're taking the natural
log of the values before plugging them into the model<br>

> 💡 Using the natural log of the features instead of the
features themselves is common `when there's reason to believe
that the relationship between the risk and the features is
linear in the natural log of the features`<br>

The **second** thing to notice here is that there's an
***Intercept feature***<br>

The value associated with the Intercept is always 1, so we
can think of the Intercept as `the expected risk score if all
of these values are 0`<br><br>

Now let's compute the MELD Score for this patient<br>

First, we enter in the natural log of the Creatinine value to
get natural log of 1, which is 0, and multiply that with the
coefficient value to fill in the `Coeff. x Value` column<br>

We repeat this with the other 2 lab features and we already
have the Intercept, so we sum up the `Coeff. x Value` column
to get a score of 1.01<br>

For convention, this score is multiplied by 10 and rounded
to get an output MELD score of 10<br>

Feature | Coeff. | Value | Coeff. x Value
------- | ------ | ----- | --------------
Ln Creatinine (mg/dL) | 0.957 | ln(1.0) | 0
Ln Bilirubin (mg/dL) | 0.378 | ln(2.0) | 0.26
Ln INR | 1.120 | ln(1.1) | 0.11
Intercept | 0.643 | 1 | 0.643
**Score (multiplied by 10)** | | | **1.01 * 10 ≈ 10**

<br>

Note that a score of 10 isn't directly telling us the
probability of survival at 3 months, but `is informative when
comparing to the MELD score of other patients`<br><br>

***
<br>

# 7. Risk of Heart Disease

In our final example, we will look at the **10-year risk of
heart disease** for patients 20 years or older, who don't a
already have heart disease<br>

```
For patients 20 or older who don't already have heart disease ==> What's the 10-year risk of heart disease?
```
<br>

This is the **Atherosclerotic Cardiovascular Disease (ASCVD)
Risk Estimator Plus**<br>

We'll once again compute the ASCVD score for a patient by
hand<br>

🧬 Here, we have 55-year-old African American non-diabetic
female nonsmoker with the following features —<br>

* `Total cholesterol` 213 mg/dL
* `HDL—C` 50 mg/dL
* `Untreated systolic BP` 120 mm Hg
<br><br>

There are a couple of new features in this model<br>

Feature | Coeff. | Value | Coefficient x Value
------- | ------ | ----- | -------------------
Ln Age (y) | 17.114 | |
Ln Total Cholesterol (mg/dL) | 0.940 | |
Ln HDL—C (mg/dL) | -18.920 | |
**Ln Age x Ln HDL—C** | 4.475 | |
Log Untreated Systolic BP (mm Hg) | 27.820 | |
Log Age x Log Untreated Systolic BP | -6.087 | |
Current Smoker (1 = Yes, 0 = No) | 0.691 | |
Diabetes (1 = Yes, 0 = No) | 0.874 | |
**Sum** | | |

<br>

**First**, notice that in addition to taking the natural log
of some of the features, `we have features which are the
product of 2 things`, like the natural log of Age multiplied
by the natural log of HDL—C<br>

These are called **Interaction terms**, and we will dive
deeper into why these are useful, but for now, all that's
necessary to understand is that for Interaction terms, we're
going to multiply 2 values<br>

In this case, we will take the natural log of 55 and multiply
that by the natural log of 50<br><br>

The **second** thing to notice here is that `we have a
negative coefficient associated with some of the features`<br>

For example, we have a negative Coefficient associated with
the natural log of HDL—C<br>

> 💡 Here, negative Coefficient means that the contribution
of this feature is negative which means that it's lowering
the score<br>

Now this makes sense because **HDL—C** is `high-density
cholesterol`, that's often called *good cholesterol* and thus
we might expect would lower the risk of heart disease<br><br>

We can similarly compute the risk score, as in the previous
examples, where we fill in the values using the information we
have<br>

Then we multiply each of the values by the Coefficient and
sum up the `Coefficient x Value` column to arrive at a sum<br>

Feature | Coeff. | Value | Coefficient x Value
------- | ------ | ----- | -------------------
Ln Age (y) | 17.114 | 4.01 | 68.58
Ln Total Cholesterol (mg/dL) | 0.940 | 5.36 | 5.04
Ln HDL—C (mg/dL) | -18.920 | 3.91 | -74.01
Ln Age x Ln HDL—C | 4.475 | 15.68 | 70.15
Log Untreated Systolic BP (mm Hg) | 27.820 | 4.79 | 133.19
Log Age x Log Untreated Systolic BP | -6.087 | 19.19 | -116.79
Current Smoker (1 = Yes, 0 = No) | 0.691 | 0 | 0
Diabetes (1 = Yes, 0 = No) | 0.874 | 0 | 0
**Sum** | | | **86.16**

<br>

In practice, this sum is converted to a risk score using the
formula below<br>

> 💡 **Risk = 1 - 0.9533<sup>e<sup>Sum - 86.61</sup></sup>**
<br>

Don't worry too much about where this formula is coming from<br>

All that's important to understand is that we can plug the
sum (86.16) into the formula above, to arrive at a 10-year
probability of getting heart disease for this patient, that
comes out to 3%<br>

**1 - 0.9533<sup>e<sup>86.61 - 86.61</sup></sup> = 3%**<br><br>

***
<br>

# 8. Risk Score Computation

In this lesson, we will **formalize the computation of
scores** that we've seen in the previous examples using risk
equations<br>

We'll also look at Interaction Terms and why they're useful<br><br>

The prognostic models we've seen so far express Score as a
sum of a feature times the Coefficient associated with that
feature<br>

For example, if we have the `Age` feature, we would multiply
it by the Coefficient associated with `Age` and repeat that
for other features like the `Blood Pressure`, & then sum up
all of the contributions to get a sum Score<br>

**Score = Age x coefficient<sub>Age</sub> + ... + BP x coefficient<sub>BP</sub>**<br>

This is what's called a **Risk Equation**<br><br>

Now, the Risk Equation doesn't need to be linear in the
features themselves, `they can be linear in the natural log or
in log base 10 of the features`, like we saw in the MELD
score<br>

**Score = ln Age x coefficient<sub>Age</sub> + ... + log BP x coefficient<sub>BP</sub>**<br><br>

Finally, `Risk Equations can also include Interaction Terms`
like we saw in the ASCVD risk example, where we might have
two features that are multiplied with each other, and then
further multiplied by the Coefficient associated with the
product<br>

**Score = ln Age x coefficient<sub>Age</sub> + ... + log BP x coefficient<sub>BP</sub> + ln Age x log BP x coefficient<sub>Age, BP</sub>**<br><br>

Let's take a deeper dive into understanding what these
Interaction Terms do<br>

To understand Interaction Terms, let's take a look at Risk
Equations without Interaction Terms first<br>

Let's say we have a risk which is dependent on the natural
log of the `Age` and the natural log of the `Blood Pressure`<br><br>

```
Without Interaction Terms
```

**Score = ln Age x coefficient<sub>Age</sub> + ln BP x coefficient<sub>BP</sub>**<br>

We can graph the risk as a function of these features with
the natural log of `Age` on the x-axis and the natural log of
`Blood Pressure` on y-axis<br>

The risk is represented by the colors where `yellow is
greater risk than green, which is greater risk than blue`<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Risk-graph-1.PNG)<br><br>

> Without Interaction Terms, if we were asked about the
relationship between `Blood Pressure` and Score, we would
say that as `Blood Pressure` or the log of `Blood Pressure`
increases, the Risk increases<br>

We can see this by simply fixing an `Age`, so we have the
yellow line fixing an `Age` and looking at how the Risk is
changing **as the log** `Blood Pressure` **is increasing**,
we can see moving from bluish green to green to light
green<br>

So **the Risk is increasing**<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Risk-graph-2.PNG)<br><br>

Now let's see what happens with Interaction Terms<br>

```
With Interaction Terms
```

Now here's the Risk Equation with Interaction Terms<br>

**Score = ln Age x coefficient<sub>Age</sub> + ln BP x coefficient<sub>BP</sub> + ln BP x ln Age x coefficient<sub>BP, Age</sub>**<br>

Notice that we now have a new term which is multiplying the
natural log of `Blood Pressure` with the natural log of `Age`
with the Coefficient associated with this product<br><br>

Graphically, we have an interesting picture going on<br>

> With Interaction Terms, if we were asked about the
relationship between `Blood Pressure` and the Risk Score,
we would say that ***it depends***<br>

Let's fix the `Age` and look at how the Risk changes when the
`Blood Pressure` changes<br>

💡 **On the left**, we first fix the natural log of the `Age`
at 3.8, and we see that the Risk is increasing as the `Blood
Pressure` is increasing<br>

💡 **But on the right**, when we fix the `Age`, we see that
the Risk is staying the same in the yellow band as we're
increasing the `Blood Pressure`<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Risk-graph-3.PNG)<br><br>

So here are the graphs without and with Interaction Terms
displayed next to each other to see how Interaction Terms can
capture a dependence between variables<br>

**Without Interaction Terms**, the effect of the 2 variables,
`Blood Pressure` and `Age` is **independent**<br>

> Whether you are young or whether you are old, `Blood
Pressure` is affecting Risk the same<br>

As `Blood Pressure` goes up, Risk is going up<br>

> But **with the Interaction Term**, `Blood Pressure` is
having less of an effect on Risk when the patient is old than
when they are young<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Risk-graph-4.PNG)<br><br>

Now that you're familiar with Prognostic Models, Risk
Equations and how Interaction Terms change the relationship
between the variables and Risk, let's look at **how we can
evaluate a Prognostic Model**<br><br>

***
<br>

# 9. Evaluating Prognostic Models

In this lesson, we'll walk through the evaluation of
Prognostic Models<br>

The **basic idea** behind evaluating a Prognostic Model is
`to see how well it performs on pairs of patients`<br>

So let's say we have a pair of patients represented by these
2 apples<br>

The apple in my left hand already looks stale and will expire
in 2 days, but the apple in my right hand will be fresh for
the next 2 days<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/apples.PNG)<br>

Now we want a **good Risk Model** to `give a higher Risk
Score to the stale apple than to the fresh one`<br>

And that's the basic idea behind the evaluation of Risk
Models<br><br>

Let's look at this more, this time with human patients, not
apple patients<br>

The **basic idea** behind evaluating the Risk Model is to
`compare the Risk Scores it assigns to pairs of
individuals`<br>

Now to evaluate these Risk Scores, we `need to know whether
the patients actually had the event`<br>

Here, we're looking at death within 10 years, so we need to
know that `Patient A` died within the next 10 years, but
`Patient B` did not<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Patients-and-Risk.PNG)<br><br>

Given this information for patients A and B, let's think
about the Risk Score that a good Prognostic Model would give
to them<br>

> 💡 A good Prognostic Model should give a higher Risk Score
to `Patient A` than to `Patient B`<br>

Now, these numbers didn't have to be between 0 and 1, they
don't have to be probabilities; all that we want from a Model
is **a higher Risk that is assigned to** `Patient A` **than
to** `Patient B`<br><br>

***

# 10. Concordant Pairs, Risk Ties, Permissible Pairs

In general, `when the patient with the worst outcome has the
higher Risk Score`, this pair is called **Concordant**<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Concordant-Pair.PNG)<br><br>

Now `when the patient with the worst outcome`, here `Patient
A`, `does not have a higher Risk Score`, the pair is called
**Not Concordant**<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Not-Concordant-Pair.PNG)<br><br>

With that information, let's try to determine whether the
following pair is Concordant<br>

We notice that the patient with the worst outcome is `Patient
C`, because they die within the next 10 years, while
`Patient D` does not<br>

And `Patient C` is the patient with the higher Risk Score,
so we find that this pair is *Concordant*<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Is-Pair-Concordant-1.PNG)<br><br>

Let's next try to determine whether the following pair is
Concordant<br>

We notice that the patient with the worst outcome, `Patient F`
has the lower Risk Score, and thus this pair is *Not Concordant*<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Is-Pair-Concordant-2.PNG)<br><br>

Finally, let's consider the case in which `we have different
outcomes, but a tie in the Risk Score`<br>

In this case, we don't consider it *Concordant* or
*Discordant*<br>
We classify these as **Risk Ties**<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Risk-Ties.PNG)<br><br>

Note that we haven't considered **Ties in the outcome**<br>

What if both patients die in 10 years or both patients don't
die in 10 years?<br>

Turns out, we `cannot use these pairs to determine who should
have a higher Risk Score`<br>

> 💡 Thus, in the evaluation of Prognostic Models, we only
consider pairs where the outcomes are different<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Ties-in-Outcome.PNG)<br><br>

A `pair where the outcomes are different` is called a
**Permissible Pair**<br>

It's with such pairs that we can evaluate Prognostic Models<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/Permissible-Pair.PNG)<br><br>

Using all of this information, we can now evaluate Prognostic
Models by giving a Score of `+1` for every **Permissible Pair
that is Concordant**, and a Score of `+0.5` for a
**Permissible Pair that is a Risk Tie**<br>

In other words, ties are given half the weight as Concordant
Pairs<br>

```
Evaluation of Prognostic Model

* +1 for a Permissible Pair that is Concordant
* +0.5 for a Permissible Pair for Risk Tie
```
<br>

***
<br>

# 11. C-Index

The **C-Index** is simply computed using this formula where
we're looking at the number of Concordant Pairs, adding that
to 0.5 times the number of Risk Ties, and dividing by the
total number of Permissible Pairs<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/C-Index.PNG)<br><br>

The C-Index has the following interpretation<br>

It tells us that if we gave a model 2 random patients A and
B, such that they have different outcomes, the outcome for `Patient A` (Y<sub>A</sub>) is greater than
the outcome for `Patient B` (Y<sub>B</sub>) and Y<sub>A</sub> = 1, Y<sub>B</sub> = 0...<br>

What is the probability that the patient with the worst
outcome, `Patient A`, gets the higher score?<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/C-Index-Interpretation.PNG)<br><br>

A **Completely Random Model** would guess correctly only
half the time and so would get a C-Index of `0.5`, while a
**Perfect Model** would guess correctly every time and get a
C-Index of `1.0`<br><br>

Let's compute the C-Index for the following example<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/C-Index-Calculation-0.PNG)<br><br>

Here we have 5 patients, let's first list out all the
**possible pairs** — `AB, AC, AD, AE, BC, BD, BE, CD, CE, DE`<br>

Let's first list out the pairs here that are **Permissible**<br>

As a reminder, Permissible means that *they have different
outcomes*<br>

**Permissible Pairs** — `AB, AE, BC, BD, CE, DE`<br>

So we have 6 Permissible Pairs, of these we'll consider
which pairs are *Concordant* and which pairs are *Risk Ties*<br>

Let's take AB<br>
For AB, we see that `Patient A` has the worst outcome and
has the higher Risk Score, so that is Concordant<br>

**Concordant Pairs** — `AB, AE, BC, BD`<br>

Patients C and E both have the same Risk Score, so it's a
Risk Tie<br>

**Risk Ties** — `CE`<br>

In `DE`, the patient with the worst outcome, `Patient D`, has
a lower Risk Score, so this is not Concordant<br>
It's not a Risk Tie either<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/C-Index-Calculation-1.PNG)<br><br>

Using this, we can now compute the C-Index using the given
formula<br>

So we have the number of Concordant Pairs as 4<br>

We have 0.5 times the number of `Risk Ties` which is 1,
divided by the total number of Permissible Pairs, which is 6,
which gives a **C-Index of 0.75**<br>

And so we have evaluated our first Prognostic Model<br>

![](https://github.com/Anacoder1/AI-for-Medicine-Specialization-deeplearning.ai/blob/master/Course%202%20%E2%80%94%20AI%20for%20Medical%20Prognosis/Week%201%20%E2%80%94%20Linear%20Prognostic%20Models/images/C-Index-Calculation-2.PNG)<br><br>

Let's next dive into real examples of Prognostic Models in
clinical practice<br>

**We'll look at Prognostic Models to determine the Risk of
Stroke, Mortality, and Heart Disease**<br><br>

***
<br>
