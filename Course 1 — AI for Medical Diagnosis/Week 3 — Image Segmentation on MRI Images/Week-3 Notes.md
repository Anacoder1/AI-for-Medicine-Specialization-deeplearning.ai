☀ 1. [Medical Image Segmentation]()<br>

☀ 2. [MRI Data and Image Registration]()<br>

☀ 3. [Segmentation]()<br>

☀ 4. [2D U-Net and 3D U-Net]()<br>

☀ 5. [Data augmentation for segmentation]()<br>

☀ 6. [Loss function for image segmentation]()<br>

☀ 7. [Different Populations and Diagnostic Technology]()<br>

☀ 8. [External validation]()<br>

☀ 9. [Measuring Patient outcomes]()<br>

☀ 10. [Congratulations!]()<br><br>

# 1. Medical Image Segmentation

This week, you'll learn about building models that are not only
able to classify whether a medical image contains a disease or not,
but also `outline which parts of the image contained the disease.`<br>

This is the task of **Image Segmentation**<br>

Image Segmentation plays a vital role in numerous medical imaging
applications such as —
1. the quantification of the size of tissues,
2. the localization of diseases, and
3.  treatment planning
<br>

We'll revisit some of the same ideas that you've learned in the
last 2 weeks and see how they extend to Image Segmentation<br><br>

In this lesson, we'll learn about **MRI data and tumor segmentation**<br>

Learning about MRI data will be important in guiding how we think
about representing data for building a segmentation model in the
next lesson<br><br>

***
<br>

# 2. MRI Data and Image Registration

In this very exciting lesson, you'll learn how you can train and
evaluate your segmentation model for identifying tumors in MRI
data<br>

You'll learn about the **U-Net architecture** and the procedure
that we can use to successfully train a segmentation model<br>

As you'll quickly see, working with 3D medical data provides
some significant challenges and you'll learn about some creative
ideas to tackle them<br><br>

We'll first start by talking about representing MRI data<br>

We want to be able to represent the MRI data in a form that we can
input into the segmentation model<br>

As you've seen, our MRI images are not just a single 2D image like
X-rays, `an MRI sequence is a 3D volume` here seen in the Axial
View [in the slide]<br>

Furthermore, an MRI example will be made up of multiple sequences,
and thus will `consist of multiple 3D volumes`<br>

```
MRI Example consists of multiple imaging sequences
```
<br>

We'll look at how we can combine these multiple 3D volumes into
one 3D volume<br>

To do so, let's pick a slice through the brain<br>

[In the slide] This is a slice through the brain viewed on 3
different MRI sequences (**Pick a Slice**)<br>

The key idea that we will use to combine the information from
different sequences is to treat them as different channels
(**Different Channels**)<br>

We can say one of the channels is red, one is green, and one is
a blue channel in the same way that we have 3 channels of an RGB
image<br>

The analogy of the 3 channels representing the RGB channels is
mostly useful for us to visualize how we're going to combine the
channels<br>

There's nothing special about the number 3<br>

The idea of using different channels also extends to when we may
have 4 or 5 sequences, which can be represented with 4 or 5
channels<br><br>

Once each sequence is represented with the different channel,
what we do now is combine the sequences together to produce one
image, which is the combination of all of the sequences
(**Combine Channels**)<br>

To us, this is now an RGB image<br>
To the machine, these are `channels stacked in the depth dimension`<br>

One challenge with combining these sequences is that they may not
be aligned with each other (**Misaligned**)<br>

For instance, if a patient moves between the acquiring of each of
these sequences, their head might be tilted in one sequence
compared with the others<br>

If the images are not aligned with each other when we combine
them, the brain region at one location in the red channel does  not correspond to the same location in the green or the blue
channels<br>

So how do we fix this alignment problem?<br><br>

A preprocessing approach that's often used to fix this is called
**Image Registration**<br>

The basic idea with image registration is to `transform the images
so that they're aligned or registered to each other`<br>

We won't go into Image Registration in further detail<br>

In the assignment, you'll work with images that have already been
registered<br>

However, it is useful to be aware of Image Registration as a
useful tool when trying to combine 3D volumes<br><br>

***
<br>

# 3. Segmentation

Once the image has been registered, we're able to combine the
different sequences<br>

We've seen how we can apply this technique to a single slice of
the brain<br>

The same idea can be extended without further complexity to all
slices (**Apply to all slices**)<br>

Now there is one volume with multiple channels that has the
combined information of many different sequences<br><br>

Now that we've seen how we can represent MRI data, let's turn to
the task of brain tumor segmentation<br>

**Segmentation** is the `process of defining the boundaries of
various tissues`<br>

In this case, we're trying to define the boundaries of tumors<br>

We can also think of segmentation as `the task of determining the
class of every point in the 3D volume`<br>

These points in 2D space are called *pixels* and in 3D space are
called ***voxels***<br>

Let's discuss 2 approaches to segmentation with MRI data<br>

The first is a 2D approach and the second is a 3D approach<br><br>

In the **2D Approach**, <br>

1. We break up the 3D MRI volume we've built  into many 2D slices<br>

2. Each one of these slices is passed into a segmentation model
which outputs the segmentation for that slice<br>

3. One by one, each slice is passed through the segmentation
model in this manner to generate a segmentation for every slice<br>

4. The 2D slices can then be combined once again to form the 3D
output volume of the segmentation<br>

The `drawback` with this 2D approach is that we might lose
important 3D context when using this approach<br>

For instance, if there is a tumor in one slice, there is likely
to be a tumor in the slices right adjacent to it<br>

Since we're passing in slices one at a time into the network, the
network is not able to learn this useful context<br>

Let's see how we can tackle this with the 3D approach<br><br>

In the **3D Approach**, ideally we'd want to pass in the whole
MRI volume into the segmentation model and get out a 3D
segmentation map for the whole MRI<br>

However, the size of the MRI volume makes it impossible to pass it
in all at once into the model<br>

It would simply take too much memory and computation<br>

So what can we do instead to still have the model be able to get
this context information in the *depth* dimension?<br><br>

In the 3D approach, <br>

1. We break up the 3D MRI volume into many 3D sub-volumes.<br>
   Each of these sub-volumes has some *width*, *height*, and
   *depth* context

2. So like in the 2D approach, we can feed in the sub-volumes now
   one at a time into the model and then aggregate them at the
   end to form a segmentation map for the whole volume

The `disadvantage` with this 3D approach is that we might still
lose important spatial context<br>

For instance, if there's a tumor in one sub-volume, there is
likely to be a tumor in the sub-volumes around it too<br>

Since we're passing in sub-volumes one at a time into the network,
the network will not be able to learn this possibly useful context<br>

The `silver lining` with the 3D approach is that we're capturing
some context in all of the *width*, *height*, and *depth*
dimensions<br>

This covers the 3D approach to segmentation<br><br>

***
<br>

# 4. 2D U-Net and 3D U-Net

Now that we've covered 2D and 3D approaches to segmentation, let's
dive into the segmentation architectures<br>

We'll start with the segmentation architecture, we would use the
2D task to build up to the segmentation architecture for the 3D
task<br><br>

## 2D U-Net

One of the most popular architectures for segmentation has been
the **U-Net**<br>

The U-Net was <code>first designed for ***biomedical image
segmentation*** and demonstrated great results on the task of
***cell tracking***</code><br>

The cool thing about the U-Net is that it can achieve relatively
good results, even with hundreds of examples<br><br>

The U-Net architecture owes its name to a U-like shape<br>

The U-Net consists of 2 paths — a Contracting Path, and an
Expanding Path<br>

The **Contracting Path** is a typical convolutional network
like used in image classification<br>

It consists of repeated application of `convolution` and `pooling`
operations<br>

[In the slide] The convolution operation here is called a `Down-convolution`<br>

The key here is that in the Contracting Path, our feature maps
get spatially smaller, which is why it's called a *contraction*<br><br>

Then, there is the Expanding Path<br>

The **Expanding Path**, in some ways is doing the opposite of the
Contracting Path<br>

It's taking our small feature maps through a series of
`up-sampling` and `Up-convolution` steps to get back to the
original size of the image<br>

It also `concatenates the up-sample representations at each step
with the corresponding feature maps` at the Contraction pathway<br>

Finally, at the last step, the architecture outputs the
probability of tumor for every pixel in the image<br><br>

The U-Net architecture can be trained on input-output pairs of
2D slices in the 2D approach<br><br>

## 3D U-Net

We've covered the U-Net for the 2D approach<br>
Let's see what we can do when we have 3D sub-volumes in the 3D
approach<br><br>

We can feed any 3D sub-volume into a segmentation architecture,
if we can replace all of the 2D operations with their 3D
counterparts<br>

This is exactly what an extension to the U-Net, called the
**3D U-Net** does<br>

The 2D convolutions become `3D convolutions` and the 2D pooling
layers become `3D pooling layers`<br>

It's okay if you haven't seen 3D convolutions before<br>

💡 All that's important to understand here is that the 3D U-Net
allows us to pass in 3D sub-volumes and <code>get an output for
every *voxel* in the volume specifying the probability of tumor</code><br>

The 3D U-Net can be trained on sub-volume input and outputs as
part of the 3D approach<br><br>

***
<br>

# 5. Data Augmentation for Segmentation

Now that we've talked about segmentation architectures, let's
talk about a technique that we can apply to the training of such
a model, **Data Augmentation**<br>

We saw earlier we could feed in transformations of chest X-rays
such that the classification label of each example state the
same<br>

Let's now look at how we can apply the same principle to
segmentation with a couple of key differences<br><br>

**One key difference** with Data Augmentation during Segmentation
is `we now have a segmentation output`<br>

So when we rotate an input image by 90 degrees to produce a
transformed input, we also need to rotate the output segmentations
by 90 degrees to get our transformed output segmentation<br>

The **second difference** is that `we now have 3D volumes instead
of 2D images`, so the transformations have to apply to the whole
3D volume<br><br>

With this, we almost have all of the pieces necessary to train
our brain tumor segmentation model<br>

The final thing we need to look at is the **Loss function**<br>

In our Loss function, `we want to be able to specify the error
we should assign an example`, given the *model Prediction* and
the *Ground Truth*<br><br>

Let's take a very simple example<br>

In reality, we're going to have a much higher resolution image
and we're going to be looking at a 3D volume, but our simple
2D example here will allow us to get a quick intuition<br><br>

[ **P** ]<br>

Pixel | Predicted probability of tumor
----- | ----------------------------
p<sub>1</sub> | 0.1
p<sub>2</sub> | 0.1
p<sub>3</sub> | 0.1
p<sub>4</sub> | 0.8
p<sub>5</sub> | 0.9
p<sub>6</sub> | 0.9
p<sub>7</sub> | 0.1
p<sub>8</sub> | 0.4
p<sub>9</sub> | 0.1

<br>

Here **P** represents the output of the segmentation model on 9
pixels, where at each location, we have `the predicted probability
of tumor`<br><br>

[ **G** ]<br>

Ground Truth | Value
------------ | -----
g<sub>1</sub> | 0
g<sub>2</sub> | 0
g<sub>3</sub> | 0
g<sub>4</sub> | 0
**g<sub>5</sub>** | **1**
**g<sub>6</sub>** | **1**
g<sub>7</sub> | 0
**g<sub>8</sub>** | **1**
g<sub>9</sub> | 0

<br>

**G** specifies `the Ground Truth on each of these pixel locations`<br>

3 of the 9 pixels are `Tumor`, represented as 1, and the
remaining 6 are `Normal` brain tissue, represented as 0<br><br>

***
<br>

# 6. Loss Function for Image Segmentation

i | p | g
-- | -- | --
1 | 0.1 | 0
2 | 0.1 | 0
3 | 0.1 | 0
4 | 0.8 | 0
5 | 0.9 | 1
6 | 0.9 | 1
7 | 0.1 | 0
8 | 0.4 | 1
9 | 0.1 | 0

<br>

We can represent the values of **p** and **g** over each of the
pixels in a table<br>

Here, each row of the table is a cell location, along with their
corresponding Prediction value and Ground Truth value<br>

For example, i<sub>4</sub> has a probability output of 0.8 and
the Ground Truth was 0<br>

Representing *p* and *g* in this table will allow us to more
clearly understand the Loss Function<br><br>

We'll use the **Soft Dice Loss** to optimize the segmentation
model<br>
It is a popular loss function for segmentation models<br>

The `advantage` of the Soft Dice Loss is that `it works well in
the presence of imbalanced data`<br>

This is especially important in our task of brain tumor
segmentation, where a very small fraction of the brain will be
tumor regions<br><br>

Let's look at the Soft Dice loss<br>

![]()  `Add equation here`<br>

```
Latex for reference
L(P, G) = 1 - \frac{2\sum_{i}^{n} p_{i}g_{i}}{\sum_{i}^{n} p_{i}^{2} + \sum_{i}^{n} g_{i}^{2}}
```

The Soft Dice Loss will <code>measure the error between our
prediction map, *P*, and our Ground Truth map, *G*</code><br>

The term followed by `1 —`, measures the overlap between the
Predictions and the Ground Truth, and we want this fraction to be
large<br>

In the first row, when g<sub>i</sub> = 1, then we want
p<sub>i</sub> to be close to 1, so that the numerator term is large<br>

We also want the denominator to be small, so when g<sub>i</sub> =
0, we want p<sub>i</sub> to be close to 0 , otherwise this term
would be large and the denominator would be large<br>

Now, we take 1 — (this fraction), such that a `higher Loss
corresponds to a small overlap and a low Loss corresponds to a
high overlap`<br><br>

With this intuition, let's now compute the loss for this
particular example<br>

i | p | g | p<sub>i</sub>g<sub>i</sub> | p<sub>i</sub><sup>2</sup> | g<sub>i</sub><sup>2</sup>
-- | -- | -- | -------------------------- | ------------------------- | -------------------------
1 | 0.1 | 0 | 0 | 0.01 | 0
2 | 0.1 | 0 | 0 | 0.01 | 0
3 | 0.1 | 0 | 0 | 0.01 | 0
4 | 0.8 | 0 | 0 | 0.64 | 0
5 | 0.9 | 1 | 0.9 | 0.81 | 1
6 | 0.9 | 1 | 0.9 | 0.81 | 1
7 | 0.1 | 0 | 0 | 0.01 | 0
8 | 0.4 | 1 | 0.4 | 0.16 | 1
9 | 0.1 | 0 | 0 | 0.01 | 0

<br>

To compute the numerator of the loss for example, we multiply
p<sub>i</sub> and g<sub>i</sub> element-wise to get **p<sub>i</sub>g<sub>i</sub>**<br>

To compute the denominator, we need the sum of squares of
p<sub>i</sub> and the sum of squares of g<sub>i</sub><br>

Similarly, we can compute these by squaring the p<sub>i</sub>
column to get **p<sub>i</sub><sup>2</sup>** and g<sub>i</sub>
column to get **g<sub>i</sub><sup>2</sup>**<br>

We can then sum up these columns to get the sum over all of the
pixels<br>

Sum of p<sub>i</sub>g<sub>i</sub> column = `2.2`<br>
Sum of p<sub>i</sub><sup>2</sup> column = `2.47`<br>
Sum of g<sub>i</sub><sup>2</sup> column = `3`<br>

We can plug in these values into the Soft Dice Loss for this
particular example and this comes out to `approximately 0.2`
which is the Loss with this particular Prediction and with this
particular Ground Truth for this example<br>

The model optimizes this Loss Function to get better and better
segmentations<br><br>

This completes all of the pieces we need to be able to train our
brain tumor segmentation model<br>

We'll look at the **Evaluation of the Segmentation Model** next
<br><br>

***
<br>

# 7. Different Populations and Diagnostic Technology

Now that you've learnt about classification and segmentation
models for medical imaging, & you've built your chest X-ray
classification model earlier, you might wonder why these
systems aren't in use today at hospitals or clinics around us.<br>

In this lesson, you will learn about some of the challenges and
opportunities to make these systems part of routine medical
practice<br><br>

One of the main challenges with applying AI algorithms in the
clinic is **achieving reliable generalization**<br>

Generalization can be hard due to a variety of reasons<br>

Let's look at an example<br>

Let's say we developed our chest X-ray model on US data and
wanted to apply it to a hospital in a different country, say
India<br>

In India, `the patient population might have X-rays that looked
different than what the model has been trained on`<br>

As a very concrete example, tuberculosis is quite prevalent in
India, but unlikely to be as prevalent in the hospitals where
we've trained our model in the US<br>

Before applying the model in India, we'd want to test the model
on its ability to detect TB or tuberculosis<br><br>

Let's look at another example, this time with our model for
brain tumor segmentation<br>

We've been able to measure the model performance on data collected
from a few countries over a few years, but MRI technology is not
standard across the globe and across time<br>

The latest scanners have much higher resolution than older
scanners<br>

Before we apply the segmentation model in a new hospital, `we'd
want to make sure that the model is able to generalize to the
resolution of the scanner at the hospital`<br><br>

***
<br>

# 8. External validation

To be able to measure the generalization of a model on a
population that it hasn't seen, `we want to be able to evaluate
on a test set from the new population`<br>

This is called **External Validation**<br>

External Validation can be contrasted with **Internal
Validation**, when `the test set is drawn from the same
distribution as the training set for the model`<br>

And if we find that we're not generalizing to the new population,
then we could get a few more samples from the new population to
create a small Training and Validation set & then **fine-tune the
model on this new data**<br><br>

All of these studies that we've talked about worked with
**Retrospective (Historical) Data**, meaning that `they used
historically labeled data to train and test algorithms`<br>

However, to understand the utility of AI models in the real-world,
they need to be applied to **Prospective (Real-World) Data**<br>

For a chest X-ray model, this might mean that we apply a trained
model to interpret chest X-rays as they're being taken for new
patients<br><br>

Why might the result of the model look different on Prospective
Data?<br>

One reason for this is that with Retrospective Data, there are
often steps taken to process and clean the data, but in the
Real-World, the model has to work with the raw data<br>

```
Retrospective (Historical) Data

Raw Dataset ==> Cleaned Dataset ==> [Training, Validation and Test Sets]


Real-World / Prospective Data

Raw Dataset
```
<br>

As a concrete example of this, the dataset you've trained your
model on has been filtered to only include **Frontal X-rays**,
where the X-ray is taken from the front or the back of the
patient<br>

However, in the Real-World, it is also common to take a fraction
of the X-rays from the side of the patient<br>

These are called **Lateral X-rays**<br><br>

💡 Before our model can be applied in the Real-World, we would
`either need to filter out such chest X-rays or tune the model to
work on them`<br><br>

***
<br>

# 9. Measuring Patient Outcomes

Another challenge for the Real-World deployment of AI models is
that `we need metrics to reflect clinical application`<br>

We've seen some ways of evaluating models we've built, which
included looking at the **area under the ROC curve (AUROC)**
or looking at the **Soft Dice Score**<br>

In the Real-World however, we want to be able to look at the
effect of our model on real patients<br>

Specifically, we care about measuring whether the model eventually
improves patient health outcomes<br><br>

One approach to this challenge includes **Decision Curve
Analysis**, which `can help quantify the net benefit of using a
model to guide patient care`<br>

Another approach is to see what happens in the setting of a
**Randomized Control Trial**, where we `compare patient outcomes
for patients on who the AI algorithm is applied versus those on
who the AI algorithm is not applied`<br>

We will look at the setup of Randomized Control Trials in
Course-3<br><br>

In the Real-World, we `would want to analyze the effect of the
model not just overall, but also on subgroups of the population`<br>

This would include patients of different ages, sex, and
socioeconomic status<br>

💡 This allows us to find key *algorithmic blind spots* or
*unintended biases*<br>

For example, skin cancer classifiers that can achieve a
performance level comparable to dermatologists when used on
light skinned patients have previously underperformed on images
of darker skin tones<br>

**Algorithmic bias** is an important and open area of research<br>

In Course-2 on Medical Prognosis, we'll walk through one scenario,
in which something as simple as `misusing missing data can lead to
biased models`<br><br>

Finally, one of the major challenges and opportunities to
applying AI medical models in the Real-World is achieving a
better understanding of how these algorithms will interact with
the decision-making of clinicians<br>

One of the **limitations of AI models**, even the ones we've
looked at in this course, is that `it is difficult and often
impossible to understand the inner workings of models to
understand how and why they make a certain decision`<br>

In Course-3, we'll look at how we can go about interpreting the
chest X-ray model we've built, and a few different ways of
getting insight into the clinical decision-making process of
these algorithms (**Model Interpretation**)<br><br>

Congratulations on completing this week on building medical
image segmentation models for medicine!<br>

You've learned about MRI data, deep learning architectures and
Loss Functions for Segmentation and evaluation methodologies<br>

In this week's assignment, you will get to see how all these
ideas go into building and evaluating your own model for
segmenting brain tumors<br>

Finally, you are now also aware of some of the challenges and
opportunities that are associated with making these models part of
clinical care<br><br>

***
<br>

# 10. Congratulations!

Congratulations on completing the course!<br>

In this course, you've learnt so much about the **intersection of
AI and medical diagnosis** — one of the key pillars for this AI for Medicine Specialization<br><br>

You've gone over some of the practical data and modeling
challenges associated with building deep learning models for
medical image interpretation<br>

You've built your own model for classifying diseases and chest
X-rays & a model for segmenting tumors in MRIs<br>

I hope that with this new knowledge and skill set, you will work
on the many problems in medicine where you can contribute
towards improving patient health<br><br>

I'm also excited for you to take `Course-2`, where you'll learn
about the application of **AI to Medical Prognosis**, where you'll
learn how to build ML models that are predicting the future of
a patient's health<br>

Then in `Course-3`, you will learn about the **application of AI
to Medical Treatment and Discovery**<br>

With what you've accomplished in Course-1, you are very
well-equipped to learn about these very exciting application
areas in the rest of the Specialization<br>

Congratulations once again!<br><br>

***
