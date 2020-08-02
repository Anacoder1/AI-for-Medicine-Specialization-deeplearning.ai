☀ 1. [Survival Models]()<br>

☀ 2. [Survival Function]()<br>

☀ 3. [Valid Survival Functions]()<br>

☀ 4. [Collecting Time Data]()<br>

☀ 5. [When a stroke is not observed]()<br>

☀ 6. [Heart Attack Data]()<br>

☀ 7. [Right Censoring]()<br>

☀ 8. [Estimating the Survival Function]()<br>

☀ 9. [Died Immediately, or Never Die]()<br>

☀ 10. [Somewhere In-Between]()<br>

☀ 11. [Using Censored Data]()<br>

☀ 12. [Chain Rule of Conditional Probability]()<br>

☀ 13. [Deriving Survival]()<br>

☀ 14. [Calculating Probabilities from the Data]()<br>

☀ 15. [Comparing Estimates]()<br>

☀ 16. [Kaplan Meier Estimate]()<br><br>

# 1. Survival Models

This week, we will talk about **Survival Models**<br>

Survival Models are a `special kind of model where we care about
the time to the occurrence of an event`, such as the time from
treatment to recurrence, or the time from diagnosis to death<br>

This is a common question that doctors want to answer for their
patients such as, *How likely am I to survive the next 5 years
or the next 10 years?*<br>

We'll look at how Survival Models are an ***extension of the
Prognostic Models*** we've already looked at<br>

Finally, we will look at how we can use survival data to build
these Survival Models<br>

Let's get started<br><br>

***
<br>

# 2. Survival Function

Previously, we've looked at the question of *what is the
probability of dying in the next 10 years?*<br>

In this lesson, we'll look at extending that question to the more
general question of *what is the probability of survival past
any time (t)?*<br>

We'll look at the **Survival Function**, a key tool in survival
analysis, which will help us answer this question<br><br>

In this lesson, we'll chat about Survival Models<br>

So in the prognostic setup we've looked at thus far, we've asked
questions of the form, *what is the probability of death for a
patient in 5 years?*<br>

So we built a model that says in 5 years, what is the probability
of death<br>

Now that is very closely related to another question which is,
*what is the probability of survival past 5 years?*<br>

This can be simply computed from this first quantity over here<br>

![](/images/Survival-Models-1.png)<br><br>

So let's say our probability of death in 5 years was 0.2 or 20%,
then the probability of survival past 5 years is simply
`1 - Probability(death in 5 years)` = 1 - 0.2 = 0.8, or 80%<br>

![](/images/Survival-Models-2.png)<br><br>

And so we asked questions of this form thus far<br>

Survival Models are going to extend that question to say, ***what
is the probability of survival past any time (t)?***<br>

This is not just 5 years, but any time *t* could be 1 year from
now or 10 years from now<br>

![](/images/Survival-Models-3.png)<br><br>

The key trick here is that earlier if we wanted to build a model
for a different time horizon, let's say we didn't care about 5
years anymore, but we cared about 10 years or 1 year, `we'd have
to build 3 separate models for the patient`<br>

![](/images/Survival-Models-4.png)<br><br>

But the trick to Survival Models is, survival models can
answer questions of the form, ***what is the probability that the
time to death is greater than 2 years?***<br>

So we can see for this patient, the probability that their time
to death is greater than 2 years is 0.8 as outputted by the
Survival Model, for 5 years it's 0.7, and for 10 years it's 0.5,
all outputted by this one single model<br>

The **key quantity** here is *what is the probability that the
time to death is greater than some number of years?*<br>

![](/images/Survival-Models-5.png)<br><br>

This is what we care about in Survival Models, the probability
that time is greater than some quantity, and this is called the
**Survival Function**<br>

The Survival Function is a function that's defined for every time
point *(t)*, and we'll talk a little more about the properties
of the Survival Function shortly<br>

![](/images/Survival-Models-6.png)<br><br>

***
<br>

# 3. Valid Survival Functions

So we have a Survival Model which is looking at a patient and
outputs the Survival Function, which is ***the probability that
the time to an event is greater than t***, where *t* could take
on any value like 1 year, 5 years, 10 years, etc.<br>

![](/images/Survival-Models-7.png)<br><br>

Now, we can graphically represent the Survival Function, where
on the x-axis we'll have `Time` which could be days, months, or
years (let's say in this case it's *months*) & on the y-axis
we'll have the `Survival Probability` at that time *t* — *S(t)*<br>

![](/images/Survival-Models-8.png)<br><br>

Let's talk about a few `properties of the Survival Function`<br>

1. The first property is that, ***the Survival at any time (u)
   is going to be ≤ Survival at any time (v), if u ≥ v***<br>

   Let's first pick *u* = 80, and we know 80 ≥ 20.<br>
   From the graph, we see that S(80) = 0.17 and S(20) = 0.89<br>

   So this holds true, we see with this example, and what this
   translates into is that as time goes on, as time increases,
   the Survival probability should never go up — it can either
   stay the same or it can go down<br>

   ![](/images/Survival-Function-1.png)<br><br>

2. The second property is that ***the Survival Function usually
   starts off with a Survival probability of 1***<br>

   (There are cases where it doesn't, but we'll mainly focus on
   this)<br>

   As time extends on to infinity, our Survival probability
   should get to 0<br>

   So our Survival Function should start at a probability of 1
   and eventually get to 0<br>

   ![](/images/Survival-Function-2.png)<br><br>

So with that knowledge in mind, let's check out which of these
are valid Survival Functions<br>

Remember the 2 properties of the Survival Function —<br>

1. As our time goes on, we'll have a decreasing Survival
   probability<br>

2. It usually starts with a Survival probability of 1
<br>

In the first plot, we see that as the time goes on, our
Survival probability is actually increasing, so this cannot be a
valid Survival Function<br>

The second plot doesn't fulfill our second property, which is
that it should start at 1, so this one isn't a valid Survival
function<br>

The third Survival Function is going down as time is going on,
and starts at 1, so this seems to be a valid Survival Function<br>

The fourth one also starts at 1, seems to be going down, notice
that here it gets to 0<br>

Now this might initially seem like not a Survival Function
because it doesn't wait till infinity to get to 0, but that's all
right — It will stay at 0 as time goes on, and so this is also a
valid Survival Function<br>

![](/images/Valid-Survival-Functions.png)<br><br>

***
<br>

# 4. Collecting Time Data

In this lesson, we'll talk about Survival Data<br>

In order to be able to model Survival, we need to be able to
represent the data in a form which we can process<br>

The primary challenge is ***censored observations***, which is
a particular form of missing data that we will look at next<br><br>

So in this lesson, we'll be chatting about Survival Data and
about Censoring<br>

Now previously when we looked at Prognostic Models where we
ask the question, *what is the probability of survival past 5
years?*, we had data which looked like the following format,
where we had a bunch of patients and for each patient we had
their outcome information which said *did I have an event, or
did I not?*<br>

So previously we `used 1 for patients that had an event` and` 0
for those that did not have an event`, here presumably in 5
years<br>

That was the data that we used and this was the outcome data that
we were using<br>

Notice the **key thing** here was that the answers we needed,
***the outcome we needed, were basically Yes or No*** — they were
ones or zeroes that specified whether there was an event that
happened in 5 years<br>

![](/images/Previously-with-Prognostic-Models.png)<br><br>

But now, when we're dealing with Survival Data, we want to
answer a different question<br>

We want to answer the question, ***what is the probability of
Survival not just past 5 years, but past any time (t)?***<br>

To be able to do that, we're going to need one information<br>

![](/images/Survival-Data-1.png)<br><br>

Let's look at how we can acquire that one information for
patients with an example<br><br>

***
<br>

# 5. When a stroke is not observed

For the example, we're going to be looking at patients who got a
treatment and we were monitoring them for a stroke event, and
seeing how much time passed between the treatment till when
they got the stroke (if they had a stroke)<br>

![](/images/Survival-Data-2.png)<br><br>

And so we have our first patient who got their treatment in
September 2018 and they had a stroke exactly after a year in
September 2019<br>

For this patient, let's say we were tracking the number of months,
this would be equivalent to 12 months since the time of treatment
to having a stroke event<br>

![](/images/Survival-Data-3.png)<br><br>

And so we would enter in this table the number of months that
have passed, that is 12<br>

![](/images/Survival-Data-4.png)<br><br>

Let's look at another patient — they had their treatment in
August 2018 and we tracked them for over a year until in October
we decided that our study was going to end<br>

We did not observe any stroke event during this time period of
14 months<br>

Since we know that the patient didn't have an event in 14 months,
we know that if they had an event it would have been after 14
months, so we write a *14+* here<br>

![](/images/Survival-Data-5.png)<br><br>

For our third example, this patient had a treatment in August
2018 and just 3 months later in November 2018 decided that they
have to withdraw from the study<br>

This happens very commonly for a multitude of reasons — Let's
say this patient had to switch countries and therefore had to
drop out of the study<br>

So we know that between August and November 2018, they did not
have a stroke, but we don't know what happened after that, so
we're going to say this patient's time is *3+* months<br>

![](/images/Survival-Data-6.png)<br><br>

We enter this into a table<br>

Now the second and third case is what we call **censoring**<br>

Censoring is an important part of Survival Data which needs to be
accounted for and something we see all the time where `we're
trying to see the time to a particular event happening, but we
might not see that event happening for one of a few reasons`<br>

We'll talk about those reasons in a bit<br>

For now, all that's important to understand is that there is this
censoring observation that we'll have in Survival Data<br>

![](/images/Survival-Data-7.png)<br><br>

***
<br>

# 6. Heart Attack Data

So let's apply that knowledge to an example<br>

In this example, `we're going to be looking at patients who get
surgery, and whether they have a heart attack following the
surgery`<br>

Here we have 3 patients, and we started our study in January
2015, and we ended our study in July 2019<br>

So patients came in at different times where they got their
surgery, and we tracked their time to getting a heart attack<br><br>

The first patient had their surgery in March 2016, and had a
heart attack in March 2017, so we can write that down as
`Patient 1 had a time of 12 months`<br>

Then we have patient 2 here, they had their surgery in July 2015,
and until July 2019 and until July 2019 we did not observe any
heart attack event<br>

This is a course of 4 years, which is equivalent to 48 months<br>

Notice that we have not seen an event in 48 months, so we're
going to put a *+* here  — thus `Patient 2 had a time of 48+
months`<br>

The third patient had their surgery in November 2015, dropped out
in November 2017, so we observed them for a period of 2 years or
24 months, and we saw that they did not have a event in that
amount of time so we write a *24+* — `Patient 3 had a time of
24+ months`<br>

So in this way, we can represent our Survival Data in the
following form<br>

![](/images/Survival-Data-8.png)<br><br>

To summarize, we had Survival Data where we made the transition
from representing our data as ***Yes or No*** like we did in the
binary setup, to asking the question ***When*** and `representing
the time from an origin to an event, also having these censored
observations as part of our data` which we'll look into shortly<br><br>

![](/images/Survival-Data-9.png)<br><br>

***
<br>

# 7. Right Censoring

So we briefly touched on Censoring earlier when we saw an
example of a patient who had a surgery and then withdrew from
the study before it's end<br>

What we could derive from this is that the patient did not have
an event in 3 months<br>

![](/images/Censoring-1.png)<br><br>

Now, it's possible that the patient went on to have an event in
Jan 2019, let's say, and earlier we looked at the event of a
stroke but this can be any event<br>

![](/images/Right-Censoring-1.png)<br><br>

But it's also possible that the patient went on to never have
the event at all and just be healthy throughout<br>

![](/images/Right-Censoring-2.png)<br><br>

We can capture both these scenarios under this broad umbrella
of **noticing that the event always happens after the last
contact, if it happens at all**<br>

![](/images/Right-Censoring-3.png)<br><br>

This is called **Right Censoring** — `the time to an event is
only known to exceed a certain value`<br>

So if we had August 2018 here and November 2018 here, then the
certain value that the time to event will exceed will be 3
months, so we write this data point as *3+*<br>

![](/images/Right-Censoring-4.png)<br><br>

There are 2 types of Right Censoring that we've looked at — <br>

1. The first in which we had `a patient who was censored because
   our study ended`, this is called **End-of-study Censoring**

2. The second type where we `had a patient drop out before the    
   end of the study`, this is called **Lost-to-follow-up
   Censoring**<br>

Censoring is a very important concept in Survival Data and one
that's very necessary to understand to be able to build Survival
Models that we'll look at soon<br>

![](/images/Right-Censoring-5.png)<br><br>

***
<br>

# 8. Estimating the Survival Function

In this lesson we'll look at estimating the Survival Function<br>

So recall that we've seen Survival Models where we have patients,
for those patients the Survival Model will tell us the Survival
Function, *S(t)*, which can be graphically shown as the
following, where `for each point in time, we can tell what the
Survival probability is`<br>

![](/images/Estimating-Survival-Function-1.png)<br><br>

The **Survival probability**, remember, is just the probability
that the time to an event is after some time *t*, where *t* can
be 1 year, 5 years, 10 years, whatever time we care about<br>

We've looked at **Survival Data**, where we
have a set of patients for who we have the outcome information
collected in the form of ***When*** — so what was the time to an
event?<br>

Here we have some observations where we observed the event,
and some where we don't observe the event, which we've looked at
as being our **Right-Censored data** and we have a few of those
Right-Censored observations here<br>

So we're going to use information that looks of this form to
estimate a Survival Function<br>

![](/images/Estimating-Survival-Function-2.png)<br><br>

We're not building individualized Survival Functions right now
per patient, we'll look at how to do that later, but right now
for any person in the population, we're going to apply the same
Survival Curve that's going to say for every time *t*, what is
the probability of Survival<br>

We can read that off from the graph, just one Survival Function
that we apply to the whole population<br>

![](/images/Estimating-Survival-Function-3.png)<br><br>

***
<br>

# 9. Died Immediately, or Never Die

And so we're **going to try to model what is the probability of
Survival to *t* months**<br>

For our example, we're going to fix *t*, let's say *t* = 25
months, and we're going to look at *S(25)* which is simply the
probability that the time to an event happens after 25 months<br>

This is the probability of Survival to 25 months<br>

We can estimate this from the data by taking the fraction of
patients that survived to 25 months divided by number of patients
we have in our data<br>

![](/images/Estimating-Survival-Function-4.png)<br><br>

Let's look at the number that survived to 25 months — Patient 3,
Patient 6 (the way we know this is because their event times
are > 25)<br><br>

Unfortunately, the thing that's making this estimate complicated
is that we have these **Right-Censored observations** — the
problem with them is that we don't know when they had an event,
if they had an event at all, we have this lower bound on the time
to the event<br>

And so we have to make an assumption here, let's try out both
the assumptions<br>

1. The **first assumption that we can make for the censored
   observations is that they die immediately**<br>

If we have someone who is censored at 8 months, let's assume
right after we had contact with them they die<br>

The number survived to 25 months in that case is only going to
include patients 3 and 6, and our denominator is going to be all
of the patients, that is 7 — this estimate comes out to around
`0.29`<br>

So this is the first assumption where we assume that everyone
who is censored just dies immediately<br><br>

2. The **second assumption we can make is that everyone who's
   censored never dies**<br>

So patients 2, 5, and 7 just live on forever, definitely beyond
25 months<br>

We now have 3 other patients who make it past 25 along with the
2 original patients<br>

Our updated estimate is 5 patients over 7 patients, which comes
out to a probability of `0.71`<br><br>

> 💡 Notice that if we assume that these censored observations
all die immediately, the Survival probability is much smaller
than if we assume that they never die<br>

Now we know that the reality of the matter is somewhere in
between these 2 values<br>

![](/images/Estimating-Survival-Function-5.png)<br><br>

***
<br>

# 10. Somewhere In-Between

Let's assume that we could call up these patients and we could
somehow find out what happened to the Censored observations<br>

So we make some calls and we can get the real value for these
patients, we see that 2 of them are greater than 25, one of them
is smaller than 25<br>

We can now have a new estimate of our Survival at 25 which
includes this real information, and this is going to have the
number of patients who survived to 25 — Patients 2, 3, 5 and 6,
so that's `4 patients out of 7 in total`<br>

Notice that our previous numbers that we had were `2/7` if we
assumed everyone died immediately and `5/7` if we assumed that
the Censored observations never experienced the event<br>

The reality seems to be somewhere in-between those numbers, where
2 out of the 3 patients survived beyond 25 and the 1 other didn't<br>

![](/images/Somewhere-in-Between-1.png)<br><br>

This is a problem that shows up all the time and we unfortunately
aren't always able to acquire the reality for the Censored
observations<br>

In that case, we face a very important question that we have to
answer with Survival Models — **How to estimate the probability
of Survival past some time *t*, here 25 months, when we have
Censored observations without actually making calls to find out
what the real event time was?**<br>

This is what we'll look at next<br>

![](/images/Somewhere-in-Between-2.png)<br><br>

***
<br>

# 11. Using Censored Data

So we're going to **estimate the probability of Survival past 25
months with Censored observations**<br>

We'll start off with a number line where we have time points from
0, 1, all the way to 25 and beyond<br>

Notice that we're considering `discrete points in time`, such
that our events are happening at either 1, or 2 , or 3, they're
not happening in between<br>

We'll start by estimating the probability of Survival past 25
months with Censored observations by writing down what this means<br>

We have `S(25)` which is the probability that our time to an
event is after 25 (`P(T > 25)`)<br>

Notice that if an event happens after 25, that means it happens
after or at 26, so we can have that modification that we make,
and we'll see why this is useful in just a second<br>

`S(25) = P(T ≥ 26)`<br><br>

Now, the probability that the time to the event is after or at
26 months, we're going to expand this out<br>

What does it mean for the time to an event to be after or at 26?<br>

It means the same thing as the time to an event happened after
26 and after 25, and after 24, all the way to it happening
right after or at 0<br>

The reason we can do this is because `T ≥ 26` implies
`T ≥ 26, T ≥ 25, T ≥ 24, ..., T ≥ 0`<br>

So we know that the time to an event, if it happens after or at
26, all the following (`T ≥ 26, T ≥ 25, T ≥ 24, ..., T ≥ 0`) must hold
true<br>

![](/images/Using-Censored-Data-1.png)<br><br>

***
<br>

# 12. Chain Rule of Conditional Probability

So we have `S(25) = Pr(T ≥ 26, T ≥ 25, T ≥ 24, ..., T ≥ 0)`<br>

The reason we got it to this stage is now we can use the
**Chain Rule of Conditional Probability** to break this down even
further<br>

>💡 The Rule says, the probability of 2 events A and B occurring
is the probability of A occurring given B times the probability
of B occurring<br>

`P(A, B) = P(A | B) x P(B)`<br>

This can be expanded out to multiple events A, B and C, this is
going to come out to the probability of A given (B, C) times the
probability of B given C times the probability of C<br>

`P(A, B, C) = P(A | B, C) x P(B | C) x P(C)`<br>

If you want some more review on this, we have some links where
you can access and learn more about this Chain Rule of Conditional
Probability<br>

But for now, we just want to know that this exists and we can
apply it to this formulation right here to break it down<br>

The way we'll break it down is to say
`Pr(T ≥ 26, T ≥ 25, T ≥ 24, ..., T ≥ 0)` is the same as the
probability that `Pr(T ≥ 26 | T ≥ 25) x Pr(T ≥ 25 | T ≥ 24)... x P(T ≥ 1 | T ≥ 0) x P(T ≥ 0)`<br>

We've just applied the Chain Rule of Conditional Probability to
simplify the former expression into this one<br>

Now we know that all events happened at or after 0, so `P(T ≥ 0)`
is just going to be equal to 1, and we don't really need to
include this term in our multiplication<br>

![](/images/Chain-Rule-of-Conditional-Probability-1.png)<br><br>

***
<br>

# 13. Deriving Survival

So we've represented the Survival probability in this following
expression that's saying, the probability that a patient survives
to 25 (`S(25)`) is the product of a bunch of probabilities which
says if he/she gets to 25 what is the probability that they get
to 26,multiplied by what's the probability that they get to 25
given they get to 24, so on and so forth until we get to the
start of time where we're asking the question if we survive 0
what is the probability that we survive 1?<br>

We're making these little steps on this timeline to get up to the
Survival probability we care about here, which is 25<br>

We're going to modify this expression a little more by
understanding how `Pr(T ≥ 26 | T ≥ 25)` can break down even
further<br>

So we have the probability that T ≥ 26 — the probability that
T ≥ 26 is also the probability that T ≥ 25<br>

We can write this down as **the probability that this person has
the event after 25 given that they had the event at or after 25**
— `Pr(T > 25 | T ≥ 25)`<br>

Notice that this is simply 1 minus the probability that the event
happened exactly at time 25 months, given it happened at or after
25 months (`1 - Pr(T = 25 | T ≥ 25)`)<br><br>

Let's look at what's happening here<br>

We're looking at the probability that the event happens after
25 months given it happens at or after 25 months, which is going
to be 1 minus the probability that it happens at 25 months, given
it happens at or after 25 months<br>

We're going to use this to extend `Pr(T ≥ 26 | T ≥ 25)` a little
further to say `S(25)` is going to be 1 minus the probability of
having an event at 25 months given it happens at or after 25
months — (`1 - Pr(T = 25 | T ≥ 25)`); this is going to be the first term<br>

The second term is going to be 1 minus the probability that it
happens at 24 months given it happens at or after 24 months
(`1 - Pr(T = 24 | T ≥ 24)`)...<br>

...and so on and so forth until we get to our last term which is
going to be 1 minus the probability that it happens at 0 given it
happens at or after 0 — (`1 - Pr(T = 0 | T ≥ 0)`)<br>

We're going to see next why it is useful to represent the
Survival probability in the following way<br>

![](/images/Deriving-Survival.png)<br><br>

***
<br>

# 14. Calculating Probabilities from the Data

So we've seen how we can represent the Survival Function using
the following expression<br>

The benefit of using this expression is we can directly estimate
`Pr(T = 25 | T ≥ 25)` from the data<br>

So the probability that the time to event is at a given time
can be estimated by looking at the number of patients who died
at that time, so the number of patients who died at 25 (the
numerator)<br>

In the denominator, we're going to look at the patients who have
a time to event at or after 25, so this is going to be the number
known to survive to 25<br>

We can estimate this quantity directly from data by looking at
the number of patients who died at 25, and we see that no
patients died exactly at time 25, so our `numerator is 0`<br>

Now in the denominator, we're looking at the number of patients
known to survive to 25 — scanning through these patients, we
see Patients 3 and 6 are known to survive to 25 because they have
event times or Censoring times after 25, so the `denominator here
is 2`<br>

So we have here 1 minus 0, which is going to come out to 1 for
the first expression, and the reason for this is that we didn't
have any patients who died at 25, and this is going to be true
for most times in the expression, we're not going to have patients
that died<br>

![](/images/Calculating-Probabilities-From-Data-1.png)<br><br>

In fact, the only 2 times for which we're going to see patients
die before 25 are going to be for `Patient 1 at a time of 10` and
for `Patient 4 at a time of 20`<br>

So this expression simplifies into computing this expression at
those 2 times<br>

So we have `[1 - Pr(T = 20 | T ≥ 20)] x [1 - Pr(T = 10 | T ≥ 10)]`<br>

![](/images/Calculating-Probabilities-From-Data-2.png)<br><br>

So we have seen how we can simplify this expression to have these
2 terms, let's see how we can simplify it even further<br>

We can evaluate `Pr(T = 20 | T ≥ 20)` using data by looking at
the number of patients who died at 20 as the numerator, that's
going to be 1 (*Patient 4*), and in the denominator, the number
of patients that are known to have survived to 20 (*Patients
3, 4 and 6*), so that's 3 patients — `1/3`<br>

For `Pr(T = 10 | T ≥ 10)`, we have the number of events at time
10 as the numerator (*Patient 1*), that's 1, and we look at the
number of events at or after time 10 (*Patients 1, 3, 4, 5, 6,
and 7*), that's 6 patients — `1/6`<br>

We can complete this computation where the first part evaluates
to 2/3 and the second part to 5/6 to get us to 10/18, we can
simplify that to 5/9, which comes around `0.56`<br>

![](/images/Calculating-Probabilities-From-Data-3.png)<br><br>

***
<br>

# 15. Comparing Estimates

So we have a **new estimate of the Survival at time 25** which
comes out to `0.56`<br>

Let's compare this to the estimates we had previously made<br>

When we had made the **assumption that all patients died
immediately at their Censoring time**, we had got a low Survival
probability of `0.29`<br>

On the other extreme, if we **assumed that all of the Censored
observations continued to live to the end of time**, then we had
a Survival estimate of `0.71`<br>

We also looked at what would have happened **if we had access to
the real event times in the data**, which is not something we can
usually observe, but we had this hypothetical scenario in which
we could call up the patients and find out the event times<br>

We had a real estimate that we could compute based on that —
`0.57`<br>

We can see that our new estimate, which takes into account
Censored observations, is much closer to the real estimate than
we could get with either of our 2 extreme observations<br>

So we've seen how we can estimate Survival probability in the
presence of Censored observations<br>

![](/images/Comparing-Estimates-1.png)<br><br>

So now that we've seen how to estimate the probability of Survival
past 25 months with Censored observations, let's look at how we
can generalize that to any time <code>*t*</code><br>

Let's see if we can generalize the expression we saw earlier to
have any time <code>*t*</code>

Here, we can represent the Survival at any point <code>*t*</code>
as a product of *i* starting at time 0 and extending all the way
to the time <code>*t*</code>, then in this we're going to have
1 minus the probability that the time to an event is at the time
*i* given it's at or after *i*<br>

So this is simply writing down for any time <code>*t*</code> what
this expression would look like<br>

![](/images/Comparing-Estimates-2.png)<br><br>

Now we've seen how we can directly estimate `Pr(T = i | T ≥ i)`
from data as the number of patients who died at the time *i*
divided by the number that are known to survive to *i*<br>

We can have a shorthand notation for the numerator of
<code>d<sub>i</sub></code>, and for the denominator as
<code>n<sub>i</sub></code>, which brings us to this expression
for the Survival Function, which is the Survival at time *t*, the
product from *i* = 0 to <code>*t*</code> of 1 minus
(<code>d<sub>i</sub></code> / <code>n<sub>i</sub></code>)<br>

![](/images/Comparing-Estimates-3.png)<br><br>

***
<br>

# 16. Kaplan Meier Estimate

This is the **Kaplan Meier Estimate** which `allows us to get
a Survival Function that we estimate from a population`<br>

![](/images/Kaplan-Meier-Estimate-1.png)<br><br>

What we did was we were able to represent the population using
Survival Data from which we built the Kaplan Meier Survival Model<br>

> 💡 The key thing about this model is that the Survival Function
that we estimate is applied to all the patients in the
population, it `isn't specific to a particular patient`<br>

![](/images/Kaplan-Meier-Estimate-2.png)<br><br>

We can compute the Kaplan Meier curves, or the *Survival curves*,
for 2 different populations and compare them against each other<br>

![](/images/Kaplan-Meier-Estimate-3.png)<br><br>

For example, let's say we wanted to know the Prognosis for
patients with Stage III cancer and with Stage IV cancer, we can
build 2 separate Survival Functions using the Kaplan Meier
estimate and we can graph the Kaplan Meier estimates for the
Survival on the same plot<br>

Here in `orange`, we have the plot for the Kaplan Meier estimate
of the Survival for Stage IV and in `blue` for Stage III<br>

If you wanted to tell the Prognosis or the Survival probability
for any patient in the population, we would simply look up given
a time, let's say 50, what the Survival probability (*y-value*)
was at that point to tell the difference in the Survival
probabilities for patients between the 2 populations<br>

So we've seen how we can use the Kaplan Meier Estimate to be
able to get these Survival curves<br>

![](/images/Kaplan-Meier-Estimate-4.png)<br><br>

This week we discussed the problem of **Survival Analysis**<br>

Using Survival Modeling, we saw how we can now model Survival
up to any time *t*<br>

We looked at one of the key features of data in Survival Analysis,
**Right-Censoring** — a `form of missing data problem in which
time to the event is not observed because the study was
terminated or because the patient left the study prior to
experiencing an event`<br>

We finally looked at the **Kaplan Meier Estimator** which takes
Censored observations into account to estimate a Survival Function
for the whole population<br><br>

What we haven't looked at yet is `how we can make individualized
estimates of Survival for patients by taking into account their
particular profile`<br>

Next week, we'll look at strategies to build and evaluate such
individualized Survival Prediction Models<br>

Until then<br><br>

***
