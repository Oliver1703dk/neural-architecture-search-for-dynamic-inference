00:01
It started  now. So you already got this  main introduction.  A  quick presentation of myself as you can already see. My name is Emilien Joer.  I've recently started as an assistant professor in this area of  HCI. That will touch a lot more on later on. uh Where I've done work.

00:28
in something called neural architecture search, which is  also very likely what  you folks are going to be working on in this course.  And then I've been doing some work with data sets,  specifically for  HCI and the specific properties the data sets  need to have in this field.  And  I've been doing some work in predictive maintenance,  which is basically using AI models to be able to predict when

00:57
something is going to break so that you can intervene ahead of time  and prevent any catastrophic failures.  I've been hired to  the new STU campus in Vejle.  So unlike,  suppose all of you folks are not sitting in Odense right now.  It might make  physical meetings weekly uh a problem, but  I'm in Odense from time to time. if we...

01:25
want to have a physical meeting to show something off at some point, then I'm sure we can figure that out if we just know a little bit ahead of time. Yeah, I'm 29 years old, so I like to think still a little bit young and like to do a lot of running in my free time. And otherwise, I'm just still spending a lot of time getting settled in to my new home that I moved into for around a month ago at this point.

01:57
You want to say some more things Francesco? Yeah, just that's my name. That's my picture of it updated. As you can see, I am a postdoc here in ONCE started in November. Research is efficient machine learning or tiny ML. Start from optimization of the computation. So pruning quantization dynamic inference, is one, the other topic you have together with neural architectural search.

02:25
AI compilers for a Therogeneous hardware, which  if you want to Google it, please do it later because it's a bit complex to explain now.  It's an automated way to generate code for your hardware starting from a Python level model. So you have your neural network in Python and you want to deploy it efficiently  on your Jetson. You use a  CUDA based, NVIDIA based AI compiler to generate efficient code for  that.

02:54
So it's  a sort of compiler but for machine learning. That's  one kind of research and  even lower level RISC-V is a type of hardware  with an open instruction set. I don't know if you heard about it  and compilers in general are compilers. So I guess you know what they are.  And that's  say  high level overview of my research.  And yes, I think we can go on. I'm older. oh

03:23
31. In you are interested. I will answer so  anyway, you can meet me if you have doubts. When we have meetings anyway, I always pull in Emil because he's definitely the expert in your architectural search. So I focus more on the dynamic inference part, he focuses more on the neural architectural search.  And I think that's like the perfect, I'd say match of supervision that you can get.  But in general, I think he's definitely good enough to answer uh

03:52
about dynamic inference and as long as it's not too complex as well I can reply on stuff about neural architectural search. So anyway,  it's  fine. But yeah, we can even have physical meetings here and we can just open a call with Emil and we do them from a conference  room. it's  actually- Yeah,  sure.  I think so at least. looked at them multiple times if I didn't.

04:20
Someone will  scold me. It's  to say sorry than asking for permission, right?  Yeah,  that's it.  Let's start with actual interesting stuff.  This is interesting. I know all of you folks are not interesting, but please also tell us a little bit about  all of you folks.  Don't necessarily have to stick to the questions that I put here, but...

04:49
Just give us a little background.

04:55
Yes,  I can start.  My name is Oliver Larsen.  I'm  going to the second  semester of my masters in software engineering  at STU in Odense.  Aside from studying,  I have a  couple of jobs.  I work as an instructor in data management and  computer systems.

05:25
computer systems, so  the one half here I work at the one, the  second half here I work at the other.  And besides from that, I also work for a  professor  at SDU, Mayar,  as his research assistant, where we also  look a bit into edge eh AI and edge  inference and such.  Yeah.

05:52
I also do a little research in AI machine learning. I just got a paper accepted for IXA focusing on edge AI on a vehicle. And then besides from that, I also try to develop a startup with this guy right now.

06:11
I chose this project  because  I thought it would be interesting.  I  like working with AI and machine learning and I think it also sounded hard and I like pursuing hard things.

06:28
So  that's the reason I chose it. Besides from  that, I also do a lot of sports.  I run a lot  right now  as well.  I just stopped doing Tread Triathlon, but I also did that.

06:44
That was a long, till. Yeah, nice to meet you, Oliver. And it sounds like you already have a lot of experience in the area, so that's more than we expected, I think. I can vouch for his experience. Actually, he should be writing a paper, so I keep the meeting short so he can go back. But I can say he's good, and I know his background, so I won't ask you any additional questions right now because I know you already.

07:14
Yeah, my name is Alexander. Yeah, and I this project because we had a course in AI, but it was interesting. I want to learn more about it.

07:32
Yeah, that's about it. How much did you do in AI already? Did you work with deep learning systems, for example? No, we didn't touch deep learning. So can you define AI in this case? What do you mean by AI? Because I'm bit confused. So no deep learning, that's particular ringer. No, we just made like...

08:00
English is mostly algorithms to solve different...

08:06
routing GPS problems. Okay, can you give me a name of one of these algorithms just to figure out what we're talking about? The ASTARS and... It was very light. No, no, was just because I couldn't figure out what you looked as it wasn't like a test. It's not a test by the means we just want to know roughly what level we are at so that we can help you the best way.

08:36
Did you do anything called linear regressions or decision trees or anything  in that regard?

08:46
Yes, we had causes in that. I've heard of it before so we have had causes in that. Okay, okay, okay. Python, are your Python skills? know you are a host Python, guess. So, question about you, Alexander?

09:17
I'm calling the other libraries from Python that we do. Okay. Yes, I guess I'll be next then. Hi, my name is Chris Biddes.

09:30
Yeah, so this project,  I've had a bit of experience with the...

09:39
from my part and also  it's kind of my... guess that I'm not that familiar with so it's interesting to get more expertise in that area.

09:56
I made a project where  you wrote it on a surface and then it would say, I think it's number 7 or it's number 8. Very basic. Okay, so you already looked at it a bit.

10:11
evening let's say. Yeah okay. All right. one. Yeah the last one I think right.  Hello everyone my name is  Oliver also  Oliver Svendsen though. um I'm  yeah a student also at the second semester  at the masters  of software engineering.  I have a part-time job as a full-stack developer  here in Odense.  My experience with machine learning is probably non-existent.

10:40
And I only have like a small  experience from  the course we had beforehand  in AI. So yeah.

10:54
You hear maybe you can answer that.  So it's Corp  in Iranian machine learning models.  It's more likely network problems.

11:16
me

11:46
Hello?  Hello? I think it's better now.  It seems better.  How much did you hear?  Okay, yeah, okay.  So my name is Oliver Svensson.  I am also at the second semester of the Masters in Software Engineering.

12:07
I have a part-time job as a full stack developer here in Odense at an app company,  but I have  really no  experience in machine learning, but a little in AI from the courses we had  in my bachelor. Sounds good.  So it's sort of the same knowledge  as the others. exactly. Maybe apart from Oliver who's been doing it.

12:37
Yeah.

12:42
All right.  Thanks a bunch uh for giving us that introduction.  We also put together a few slides  that  can help introduce the project  to you.  And  please don't hesitate to ask any questions if there's anything you don't understand.  And a lot of this is probably obvious to Francesco and I because

13:10
We've been working with it  for a very long time.  But we realize it's not always so easy when you are just getting into it. anyway.

13:29
The big picture of CI is that

13:42
If

13:48
And put a big third or you maybe can't afford  to put  any large computers.

14:04
have the energy for it and so on. Wasted computers that are very, very tight,  otherwise known as  embedded systems, because they're typically embedded into other systems. And these computers, they're uh very cheap.  you can easily produce  some cheap equipment using them.  They consume low power. So  say you are  running  a system somewhere where you don't have access to the power grid, you

14:34
it.

14:39
It's...  Should we try mine? We try yours.  But if you use some low-pass  on time,  just on a battery. The downside of this is that these computers are not very powerful at all. They have a very, very small CPU,  typically just  a single core that runs.

15:09
a lot slower than the one in your...

15:13
That's up.

15:32
uh So  we have a bunch of these edge devices and they're deployed all over the world in your coffee machine, in security cameras,  on offshore windmills and so on. And in some of these places we would like to run some AI models on them.

16:01
But because they have so few resources, we really have to create some small and efficient AI models.  There are different ways to do so. You can design them  from scratch to be small and therefore not consume a lot of resources. But other things that we do is that we do some optimizations. uh Two very popular ones  are called quantization, which is where you take the

16:29
weights that  make up these AI models and you uh convert them to a different data type that is typically  using fewer bits and also support more efficient execution on hardware. So that could be taking  a model stored in 32 bit floating points and converting that into a  that's stored  in 8 bit integers thereby

16:58
roughly reducing the size of the model by four  and  making it much more efficient to execute  this model just because the way that  the computation works down at a hardware level, it is much, much faster to do multiplication and addition using integers than it is in floating points.

17:22
Another optimization technique that is extremely popular is... Wait, Daniel, we have a question. have a question.  Okay, sorry, I didn't see that. please. Yeah, sorry, sorry, I didn't really... But um in terms of this quantization,  when you reduce from four,  would that not also impact the accuracy or is it...

17:44
It does. It can impact the accuracy because you definitely don't get the same model. Yeah, exactly. I would say experimental evidence has shown that  at least down to 8-bit integers, you can actually  do  that conversion without  losing a significant amount of accuracy. That said, it is still an active research area and there's especially research being done in the area that...

18:12
maybe the top level accuracy of models  might not suffer too much that that's say the accuracy of the model on a full test set or so, but the accuracy on minority classes in the data set might suffer. So that could be  some fairness  issues  when doing this, but again, still an active research area. Okay. Yeah.

18:41
Hope that answers your question. Yeah, thank you.  Another optimization technique that we use  a lot.  Now I visualize it here using a deep neural network, since that's what a lot of people work on, but this can also be used in  issues, in for example, decision trees. But the idea is that you try to remove half of the models that do not contribute significantly to the final result.

19:10
In doing so, can  skip these computations at inference time and  thereby  both save  on the weight that you have to store in the model and the computations that you have to do  at a later time.  And that's a lot of extra complexities to that, that sometimes when you do  pruning, you have to make sure that you do it in a structured way so that  the hardware that you're running

19:40
this,  your machine learning model on can actually  take advantage of  the smaller amount of weights.  This is typically done if you have a, this is typically a problem if you have  some hardware that tries to do parallel execution. And then it expects  all of these weights to be there. And if they're not there, it results to some uh very slow uh computation where it does.

20:09
one computation at a time. So that's nothing you need to understand completely now, but if you end up doing pruning eh of your models and they end up not being a lot faster, then that could very likely be the culprit.

20:26
Just  say  if you have a question. think maybe Francesco, you can this slide.  So  just before while I was talking, I was trying to show you  microcontroller. So where we are generally deploying the neural networks. Actually, this uh is a sub-peripheral. So there are some.

20:53
Other parts are not actually necessary. actual microcontroller is just one square. Yeah.  And we're talking about if we're giving some numbers in general, 520 kilobytes  of uh RAM, think. But yeah, that's the orders of kilobytes, as Emil was saying.  So the idea is that  if you think about the Raspberry Pi, Oliver, that's considered like a canon, uh like end of the spectrum.

21:22
a very powerful edge device because it uh consumes a lot of energy. It's not lightweight.  So it's closer to a GPU than what we are working on generally. that's one of the additional complexity of working with this type of devices. But you won't bother with that.  it just for the sake of knowing, maybe when you're writing the context of your final  report or whatever.

21:51
is that uh these devices are bare metal, so they are not running  a  So that changes some of the things because  as you know,  if you don't have an operating system, you don't have dynamic memory allocation implemented. So your malloc,  if you remember something of C or  your list from Python  or your vectors that are from C++ that are resizable,  that's

22:21
not happening unless you put some additional layer that handles them for you. Just to add a little bit. But yeah, that's not fundamental for this project. The idea, as Emil was saying, that quantization will be probably something that we will touch because in the end we always quantize the model. I don't think it makes sense to leave them FP32.

22:48
And  pruning is something that probably is, you will see because it's applied in your architectural search, even if it is at the channel level most of the times, right? I think.  Yeah, we can get into a big discussion about the difference between pruning and neural architecture searches, but that's definitely some over That's part of the literature review that you will have to do.  But in general, here you see  a slide about dynamic inference.

23:18
Dynamic inference is, I guess, one of the least-known optimizations in the tiny-mail world,  because it's a bit more peculiar. Instead of being done  offline, beforehand, like on your computer, you quantize and then you export the int8,  so the quantized model, to your microcontroller.  This is something that happens at runtime, so while the model is running.

23:43
And I think Oliver, you've tried something similar to dynamic inference because you in general,  the idea is that you change the model or at least the computations at runtime depending on some external signal.  And this, in this case, you see an example that is depending on the input complexity, you go and change the model. How does it work?  So I give you an image, a crystal clear image of a cat.

24:11
and I ask you, is this a cat? Your brain will tell me immediately, yes, this is a cat, very easy. If I give you a blurred image of a cat and I ask you, is this a cat? You will have to put more effort, no?  brain computations.  And in the end, probably you will tell me, yes, this is a cat. Well, this doesn't happen for machine learning models or deep learning models.  If you give a crystal clear picture to the model or a super blurry one,

24:41
the model, you will always spend the same amount of compute. No, the same amount of operations will be performed. What dynamic inference does instead is changing the amount of computations depending  on in this case, the input complexity. So easy inputs, less compute, more energy efficient, harder inputs, the full computational budget that you have, but an accurate result. So you're not trading off.

25:11
uh too much accuracy for energy efficiency. It's still a trade-off. So here what happens is that the image is first in the big little architecture. The paper I sent you and that you probably have read already or if you didn't,  now you will know. uh The image is first processed  by the little model, an inexpensive but not so accurate neural network. You can visualize it.

25:40
as a small convolutional neural network, for instance,  a LENET  to be, uh even if it is not that efficient,  a  small model, and you get an output. The output is just  the probability  distribution of all the classes. And in this case,  at that point, you will use a policy.  In the paper I sent you, I think there is both the max policy and the... uh

26:10
the other one that is the difference between the two top classes. I blanking out, but score margin, sorry, the score margin. But in the end, will lose, how it works is that one of the probabilities is much larger than the others. If yes, it means that the model is very confident that that is the class, the correct class, So for instance,

26:36
In a classification problem, are classifying if it is a dog or a cat. If you get 99 % cat  as an output of the first model, it's very likely that is correct. So in this case, you stop the inference. You just take the output of the little model, because one class is much more likely than the others. You are confident enough that the little model did an accurate prediction. In the opposite case, so we don't go over this

27:06
user defined threshold, you run the same input over the big model. So you perform a second inference and you take the output of the big model as the final one. So you see a problem here, two problems actually. The first is that you are deploying actually two models instead of one. So in terms of memory, we talked at the beginning, ah, memory is very constrained. Yeah, that's an overhead.

27:35
This is the first overhead.  That's there. Even if there are some techniques that I'm not interested in  right now, because it would make the  architectural search bar much harder. So  this is an overhead of this approach, but you have to consider that the little model is very little. So it's very small. Its orders of magnitude generally is smaller than the big one. So the overhead in terms of storage or memory  is negligible.

28:06
that's debatable, but yeah, it's much smaller.  And in terms of computations, what happens is that the average uh number of computations goes down because most of the time the input will be easy.  So most of the time you will just need the little model. When you have a complex input,  you are actually increasing slightly the number of computations because you're running both the little model and the big one.

28:35
But that's generally acceptable because it  has to happen very rarely. And if you wonder why are we using a max policy or the score margin or something so dumb and not an  amazing neural network to decide whether to use the small or the big model, the answer is because the policy part has to be lightning fast. Otherwise, if it is too computationally intensive, uh

29:04
what you save on average, because you just run the little model, you lose because in the end you have to execute a very heavyweight policy. So that's why you will always see some very light policy to understand if the model is confident or not enough.  And this is like the summary of what you read in the paper.  The applications are various. You can go from image classifications to time series.  For this work, actually...

29:35
don't care very much about the type of data that we would like to see. Probably it would be image classification, but it can be extended to anything else. In this case, we are quite  orthogonal  to the type of data.  It would be for sure a classification, though not a regression problem. Supervised classification, just  if you want to know in advance.

30:02
Any question on this part, even if you read the paper already, so it's probably something that you had already.

30:10
Okay, so it's either very unclear or very clear.  Nothing  in  Perfect. Since it's like 50 % of the project, I will suppose that it's very clear, hopefully. Otherwise we will find out in a month or so. But yeah, so if there are no questions, think we can go to the next slide. Yeah,  I think you have to control.

30:38
Yeah, I'm on the controller. That's your part. You are the expert. Yeah, I can talk about this. Yes. So that brings us to the idea of a neural architecture search. And now we heard that a lot of you haven't heard so much about deep learning yet. So maybe the figures on the bottom left is

31:06
a little bit confusing for you folks. And are you able to see my mouse pointer when I'm moving that around?  yes. Okay. So neural networks are typically made up  of neurons such as  the one X one over here.  that  contains  a weight of several weights  and what's called the bias.

31:36
When it's in the input layer, this typically uh represents uh directly the input that is sent on  to the next layer where it's multiplied  by a weight and the same happens for each of the other inputs  and  a bias is added and then that,  the result of this computation is sent on to all

32:05
other neurons in the next layer.  That's a very typical structure of neural network.  Formally, it's perhaps called feed-forward neural networks because there are no arrows going back.  Don't worry about it. No one uses that  to my knowledge anymore.  Yes.  The problem is when you're creating these neural networks that you yourself have to define

32:35
the structure of how many layers does this neural network have, how many neurons should be in each layer,  should we just have direct connections  from every neuron in one layer to every neuron in next layer, or should we introduce these  skip connections, as we call them, where the output of a neuron from one layer actually goes to the input of a neuron in

33:04
a layer further down.  You can create a bunch of different and weird uh architectures. And it is very hard to say before you deploy the model, before you train it and you test it, which one is actually going to perform  better. So there's sort of two ways that you can approach this.  You can have an expert.

33:34
it and try to try out one model, test it,  sort of try to apply his or her domain expertise to make some changes to the model that they think are going to work, test it again  until they get a model that is  up to their standards or they've used all the time or the budget that they've been allocated to this task.

34:02
That would be the way of manually designing  these networks. uh And that can take a very long time, as I'm sure you can imagine. So as an alternative to this,  this idea of neural architecture search has been proposed, where instead of having an engineer that sits and tests through different models,  model architectures, you have a computer. uh

34:28
run through  a lot of different architectures and trying to learn which architectures work well.  And then in the end, uh find the optimal  or  apparito frontier  in case you're working with several uh competing objectives  of models that perform very well.  And that's a bunch of literature that we can dive into  at a later time on

34:57
how you do this,  just to give you a teaser, you  typically either use reinforcement learning, maybe you've heard of that, or evolutionary algorithms.  And then there's a  sort of new research area where you also attempt to  do a gradient descent  on the  neural architecture search itself.

35:24
in something that's called differentiable neural architecture search.  I think you should spend some of the early week here and trying to research that.  And if you're not too familiar with deep learning from the beginning,  maybe I would say that differentiable neural architecture search might be a bit of a big too big pill to swallow.  And you should maybe stick to some of the more traditional.

35:52
approaches in the beginning.  Do you have any questions about this?

36:00
Oliver Larsen, have a fast question. Yeah.  You said  reinforcement learning and another one and then the gradient descent on neural architecture search. What was the second one?  Evolutionary algorithms  or something called genetic algorithms is another name for think a subcategory of it. Is that something you've heard before?  No, it's not. I just didn't hear it. So I wanted to write it down.  Yeah, sure thing.

36:30
Yeah, the suggestion is finding a survey and possibly something. think we sent you one. And your link. And these contain a bunch of other things. So you can try to look there if you didn't do that already. Yeah, the idea is that, reinforcement learning, I guess you know the genetic algorithms is just trying different combinations of layers and other parts.

37:00
and converging to a better architecture. it's some sort of a realistic  approach, very slow. the differentiable one is that is uh even more complex. And so you go by gradient sense and start to, uh you train also the architecture. I wouldn't know how to explain it without any background in deep learning, but yeah.

37:30
you will read about that. It's definitely  something quite interesting.  As Emil was saying, I guess it was super clear.  It will take some time. Initially, guess here you have a very high, let's say, initial step where you will have to learn at least some basics to start and appreciate, let's say, the topics. And then you will see that  it will become easier.

38:00
But yeah, unfortunately, this is the  initial step is harder if you don't have any  background in deep learning for sure. So don't worry too much about it. It can be done a step at a time.  Right now, the most important thing is that you just understand the big picture of what is it that we are trying to  achieve  and why.  Then I think that is going to be very helpful.

38:31
Okay, can I move on to the next slide?  Yes, that's fine. And thank you for the answer. worries.  Do you want to take this Francesco? Yeah. the idea here is, you've seen what is dynamic inference. You've seen that new architectural search helps us to find the perfect model  for our task, at least its perfect configuration. This,  I would add that this new architectural search, you can also give them

39:00
not only find me the most accurate model, but also find me the most accurate model given a compute  or a memory budget. So if  I have 520 kilobytes of memory, find me the most accurate uh model  for this task. Okay, so you can also put there already in the search the memory limitations that you have. And the idea here is that

39:29
uh Okay, for dynamic inference for big little, we have two models. And right now what we are doing  is to train the little model possibly with an ass.  Then we do an ass on the big model. And then we try to merge them together to run them together.  So they are treated as separate, completely separate entities. This has its advantages because

39:59
You can optimize one and any point in the future optimize the others or first a little in future the bigger at any time you can change them and swap them and somehow it will work. But the question is this, if we are looking for the maximum efficiency, isn't it is a bit suboptimal to optimize them separately. Would it make more sense to

40:25
Already when you're running a neural architectural search, consider that you have to get two models, a little and a big one. So the idea here is that we want  a NAS that actually outputs two models, not one. One for the little  and one for the big. So it's a joint optimization in this case,  to use the scale with them. uh

40:55
The  second issue is even more complex, but this is step by step. So don't worry. These are, let's say, the open points, the complexity.  What's the research? This is the  research here.  So  you will probably be able to understand it better  once you have finished your literature search and  learning part.  And the second issue anyway is that

41:24
even if we jointly optimize these two models. So we say, stay under this budget in terms of memory and find me two models. In the end, they are not run separately. These two models, they are run in a cascade, you know, first the little and then the bigger. So can we simulate this type of execution while we are running our new architectural searcher in this way?

41:54
Ideally, what we want is that we let the new architectural search understand that we want to stay  behind a certain compute or memory budget, but also we want to convert to a little model that is very good at estimating the class probabilities because we use that as a way to understand if we have to enable the big model. So in a way, we want to uh

42:23
First extend the new architectural search to output two different models instead of one, and then extend it even further to give  this algorithm an idea of what they will be used for. So for dynamic big leader inference.

42:43
And just to be, just not to scare you to death,  it will be step by step. So we will have like, let's as far as we get,  we'll see as far as we get. The research part is this one. So the further we go, the more into the research topics we are. But if  we get only the first point with your project, it's already an interesting result.

43:13
So I wouldn't say that you have to complete all these points, solve everything here to get through this course. So don't be scared again.  But yeah, probably we will have to  go again, I guess on the project details when you are  done with your background, your upgrading in the literature and in the background, because otherwise it's just too confusing in my opinion.

43:43
right now.

43:47
I know it's very hard, but any question right now, anything that you want to clarify,  or you prefer, I understand if you prefer first to read about your architecture search, deep learning, and then ask again, it's fine.

44:05
Yeah, now maybe add  not seeing if you have any questions right now, but this is also at least the first time for me that I'm  participating  in this course and proposing projects,  perhaps the same for Francesco. if this turns out to be way too much,  it's not necessarily your fault. Maybe we  set out  a bit.

44:32
too big of a project. again, take it step by step and it's not necessarily your fault. It might as well be our fault. It's not fault. It's a research project. It's a research idea. This is a research. So if you manage to convert to something, it's fine. It may be publishable. It may go on. You can even extend it further. But for the sake of the course, that

45:02
the expectations are much, let's say, more contained.  So there is no failure here. If you came in the background,  you understand the background, is not impossible, on deep learning, on neural architectural search, and you start trying some stuff in this direction, it's already a success, in my opinion.  If we manage to get even further and propose something new, this is the research part.

45:32
and even better, but I wouldn't say we will  have the risk of failing. uh We have multiple steps. The farther you go, the more interesting it gets,  but it's not the better in terms of your final  mark, if you  want to ask about that. That's research. The further you go, the more interesting the paper  we can eventually publish.

46:01
Also, and I think it's one, I know,  I won't anticipate the next points because I've seen the slides. If you don't have any point  on this slide, we can keep going. Yeah,  can you also let me know if, oh, okay. Okay, you only have a... Yes,  I have a fast question. oh It's a question also prepared before the meeting. You mentioned something that this could also be...

46:29
publishable  if we do it correctly. So that was one of the questions just to clarify. So  we're all on the same side.  In addition to that,  I think this could require some compute resources as well  for the  training and evaluation maybe.  Do we get access to  compute resources, for example, in UCloud  or how does that work?

46:59
So that is just as much a question that I had to you folks. That's okay. That's an open point. Let's say that's two  since we are targeting very tiny devices.  Okay. ah won't need any GPU. Because in the end, what we will train is tiny models.  And if you try to train tiny model on a on a GPU, what happens is that it's slower than a CPU.

47:27
And the reason being that moving data from the CPU to the GPU, it's actually slower than the speed up that you get during the computation on the GPU, if you're interested in the technical reason. But the idea here is that you probably won't need much in terms of resources. We will see if we can provide something. But in general, a laptop is super fine to do stuff. I look into that.

47:56
to be frank, I still don't know what exactly we need.  If we instead are  considering the deployment, the final model running on a microcontroller that we can, let's say, showcase somewhere, I can provide, obviously, the microcontroller that you can use freely. So  if the problem is the training, I would say no worries. Laptop works without any  problem.

48:26
If we need more, I'll try to get more resources, but I don't think so. uh And  instead, if we, but this is not even my point. I don't really care about the deployment on microcontroller. can do it. We can see where we are. And eventually do it later.  I can provide the actual hardware to deploy. yeah,  does it answer mostly your question?  Yes, it does.

48:55
Yeah, that's fine. Thank you for the answer. But, but, yeah, so maybe a quick question before we continue. Do you have a strict back line here at in five minutes that we need to end the meeting or can we go a little bit over time? It's okay for me if it is for you. It's fine for me. It is, is, is. Yeah, I just wanted to clarify that before we start.

49:23
diving deep into the platform topic. Yeah, because I put this question here, as you can even see on the slide that perhaps you cloud could be an option and you can also find non GPU machines on you cloud that you don't have to keep your laptop awake all the time, all times to do training that that can also be quite nice. But

49:51
Honestly, I don't know too much  due to the short time that I've been at the SCU. How much students actually have access to uCloud  by default. If you don't have access by default to uCloud, I'm almost certain that  if we, Francesco and I, vouch that you should have some resources,  you will get some.  But do know yourself how much you have by default?

50:18
Yes,  we don't really have access to  any compute that could be used. We have some U1 standard H, 10,000 core hours.  I'm not sure. I haven't used that, but it's possible to invite us to  projects with  compute.  I've been there before. Okay. Well, I tried to ask around, but I didn't understand the last part.  What did you say, Oliver?

50:46
The last part or the first part? The last part. It's because on our student projects, our workspace, we don't have that much compute.  I haven't tried to use that. I tried in the beginning, but it didn't work.  But it's possible to  invite our students  to workplaces with compute. em So if you get any workplace with compute,  it's possible to invite us to a new club.  I think that's the way it should be done.

51:17
Okay, okay. I'll try asking around as well. I think I might have a good idea of how you should do it, Francesco. So you can do like applications in there and I guess we just have to write a small text about what we're trying to do and then we'll get some resources at us. We can have a chat later on, Emil, to understand how to do it. Okay, we'll see. If you folks would like us to do it, if you prefer to just work on your laptop, that is also fine.

51:46
and even easier for us. Yeah, for sure. It won't kill the resources on the laptop. We are talking about very tiny models that run like, I guess, I don't know, the data set we were talking about, the wake for your semil, with vision is quite fast, right? It's the resolution of 32. The resolution can be quite small. Yeah, you can sort of configure that as

52:15
as you wish, but it is also very large, consists of millions of images. So  if you're running through the whole thing, it can take  a very long time. you're doing neural architecture search, you're also like training several models after each other.  it can end up taking a long time. And that's why I would perhaps personally prefer that I didn't have to run that on my laptop and keep it awake at all times.

52:43
Makes sense. Makes sense. We can  figure out how to ask for some compute ourselves. We'll write you about that. That's fine.  If it isn't possible with compute, I have a spare laptop as well. So it's also a  possibility to use that.  I'm sure we'll find some resources. Yeah.  Okay. We give you access to one of...

53:11
If you see here,  we also have, guess, the option of one of our servers, but I have to understand how to do that  as well. But yeah, we can figure out something for sure. Leave that to us as an action. Yeah, that's fine. Good.  Okay, I'll try to run  somewhat fast through the remaining points since we're running out of time.  But  I had a slide here where I proposed

53:41
some starting points that is at least a good place for you to start  to get started on the project.  You can try to figure out this development environment. suppose now Francesco and I is going to  at least look into uCloud and the other servers that the group might have.  And maybe now use your laptop and when as soon as we get the resources, eventually we will.

54:10
let you run there, but for now, yeah, this experiments is a meal,  saying just start with your laptops just for now. So then if you haven't done any deep learning before, I have suggested you find this PyTorch webpage, you try to run some of the uh quick tutorials that they have there to sort of see the workflow of developing a model in PyTorch.  Then  you should

54:38
spend a little time reviewing the literature of the research field.  Francesco and I already sent you some initial uh papers  or blog posts that you can read.  But they're probably not sufficient to learn everything. So if you find something in the  text that we sent you that is  interesting, perhaps there's a reference around it and you can read that to learn even more  about

55:07
about something and then  if by the end of  sometimes during not next week, but the week after you can try to run  a basic neural architecture search  that doesn't have to generate two models or anything fancy like that. Then I think that  we already doing good progress. Yeah.

55:33
You will find definitely some libraries. would suggest even if you try,  Alexander, tried TensorFlow. I will try the TensorFlow. don't remember. ah Okay. Just use PyTorch. ah Okay. uh Don't bother with the others for now. Please use PyTorch because I guess most of the libraries also for new architectural search  are written, or at least the ones that we could think about modifying are written with PyTorch.

56:02
I torch this.  Yeah, wouldn't nothing else to add.  Please take a look at the background. Most mostly try to understand.  Again,  you don't have to become deep learning grandmasters in two weeks.  Just expect you to know what is a convolutional neural network? What is a convolution? What is  a mobile net?

56:31
for instance, which is probably one of the most common architectures for  images.  How did they get to the MobileNet architecture?  I think they use a new architecture search for the version two.  that's MobileNet v2. It's the  paper. You find it online,  read it maybe.  Then you should at least understand how a training loop works.

57:00
Again, I don't care if you know about back propagation, but at least the order of stuff that has to be called with PyTorch.  What is a batch size?  is the how to declare the data? I mean, basic PyTorch stuff without going too much in depth. Once you know that you can train your small PyTorch model by yourselves, ah then at that point we

57:29
you can start looking as well. I'm talking about the technical part. Literature review is another separate thing.  Then you can start looking at uh automatic  automated uh new architectural search with PyTorch. So you will probably go on GitHub  or wherever.  Google it, search GPD, uh ask around and get some libraries that do that for you. Try to understand how they

57:58
do it  and take a look  at how they work  in general, very high level and try to run your new architectural search for, I'd say to begin with CIFAR 10. Let's focus on CIFAR 10.  Small data set runs everywhere.  There's plenty of literature  and it's  fine.

58:21
Let's focus on convolutional neural network, specifically 2D convolutional neural networks, because we are talking about images, classification problems, since we are working on CIFAR-10. oh yeah, this is, let's say, the idea. Show us, like, try to, let's say that by not next week, the one after, you should show us uh some numbers that you can

58:51
get by training your model on CIFAR-10. So you take that design  one or two convolutional neural networks and you train them on CIFAR-10. And then you show us the results when you use a neural architectural search  on CIFAR-10. So just to have, say, three numbers, two custom uh human-made architectures uh of models.

59:20
on C410 and one generated or two generated with these libraries that you will find. Yeah. then that's the technical part. Yeah. So the important  thing here that Francesca is also mentioning that  since we don't have an unlimited amount of time  and because there are a lot of people that already have already done a lot of great work online, it's probably not worth it that you start

59:48
trying to implement a neural architecture search on your own. As Francesco says, there's plenty of good libraries already  online.  And at least for just running your first one, I think I'd use something like that. Then in the longer run, you should probably try to figure out whether or not it's possible to modify these libraries slightly to output two models, or if we need to  then transition to  creating something more.

01:00:17
a custom for that. Okay, so and in terms of let's say what  I would really love to see is first  one of the tasks that is not obvious you will find sooner or later but I'll tell you in advance is that when you are doing deep learning stuff one of the complexities is reproducibility. So remember early  to look into how to make your results reproducible using PyTorch.

01:00:47
That's one of the core points that you realize too late, generally when you are close to a paper. So look into that because I expect that if we run again those results on the human-made architectures that you will design, we get the same numbers on the test set of CIFAR-10. So please don't remember to set the random seed everywhere.

01:01:16
And then  I really want to use GitHub for, let's say, the code.  So you are free to create your,  let's say, repository and handle it  at the beginning. However, you think it's better. So create a group repository. If you think you will need multiple repositories,  create an organization and invite

01:01:44
invite both Emil and me uh to eat so you have multiple repositories so handle the version control however you want uh just handle it uh don't  save stuff somewhere so that we can reuse it uh later okay

01:02:07
So any questions?  Just on the technical part or the literature review or what you would have to use?  Okay, first read this slide, I guess. Yeah, we have this and one more slide to read on, promise. Yeah, so I just wrote down a little bit of what I think is good advice and there's probably many, many more things than what is here. But one of the most important things is probably to...

01:02:35
not be afraid to just try things out.  One of the  things that can happen a lot of times is you think something is really, really complex when you read about it and you're sort of  afraid to get started, but  just play around and see if you can make something work and that usually helps eh you get you progress faster. And then um

01:03:02
I would also be very surprised if you folks are not  using LLMs in some  way for coding or for  learning about things.  I'm not going to sit here and tell you that you shouldn't use it or  I never use it. But please at least make sure that if at least if you generate code or something like that,  that you do understand the output that it gives you.

01:03:28
It's not that doing the project and  instead of you and and of course  try to verify the information that  it gives you consider that  since we eventually want to go towards let's say a publication that's if we get there whatever everything that we do must be explainable like there must be a reason that we could write on a paper for because we did this because and should make sense and you should be able to explain it.

01:03:58
So you have to first agree with the large language model tells you, make sure that it is actually making sense and also figure out if it is just something too weird that  just proposing you to add random numbers or do some super fancy nonsensical and too convoluted stuff. Because in the end, the idea is writing the report and the report should be uh a report or paper should be formal and should be. uh

01:04:27
should really make sense. So  every decision  right now, okay, I don't really care if you find out that putting three convolutions instead of two when you design your small CNN is better, whatever.  But when we start designing the new architectural search  or  extending it  or  figure out uh how to  support this dynamic inference, that every decision must come with a.

01:04:57
good reason. Okay, something that you can really write in a paper and not be afraid that someone will tell you why this doesn't really make sense. So I don't get the reason. Okay,  this is  one of the core parts of, let's say, converging to our research work. At the end, you have to write everything. So everything must be justified.  Okay.

01:05:25
Good. And  then the next one is perhaps the most controversial one.  But  I don't know if you're sitting,  I think at least if you folks are sitting with some crazy ideas  of, let's look into something called neural architecture generation instead of neural architecture search, and you think that could be a great way to work on this project instead, then...

01:05:52
For me that's all fine.  I love it when research  is trying to do a little bit more than just the super safe  path.  But of course  it is also taking a risk when if you try something too crazy and it doesn't work out.  So um by all means try it out but also  be good at saying stop if it doesn't work.

01:06:22
and  let's make sure that we at least produce something that  we can show off at the end of the course.  Yeah, after the first part. Like for the first part, like getting to the new architectural search and trying it there, there is zero, zero fans. Don't, don't, don't go fancy. Just do what we,  because you have to understand and get the background. If you start trying stuff too early,  you misunderstand what we have to do. So first,

01:06:51
implemented, go at least to the new architectural search, try the new architectural search, understand what a standard new architectural search is. And once you have the background,  it's as Emil said, don't be afraid to  read another paper, you want to apply that idea. Yes, if it makes sense,  do. But first the background and let's say the related works and then the imagination and  new stuff part. Okay.

01:07:20
Then I would  at least fire you try to  create some sort of a time plan.  I couldn't care less what format you use or how specific it is, but at least that you start thinking about how many weeks that you have and  when you'd like to be done with A, B and C. I think that's going to be very helpful.  Not necessarily because you're going to stick exactly to that plan.

01:07:50
Never going to be like that. But at least you're going to start having an idea of whether or not you're on track or whether or not you're doing very well  or if you're falling behind and you need to do some changes. And then I would also say that start writing early as soon as you have something to write about, get it written down.  I unfortunately see a lot of projects,  not in this course because it's the first time I'm doing this, but

01:08:17
I've supervised master students and  very common pitfall is that you start  writing way too late and you don't have the time to go through getting reviews from  supervisors and other people to make sure that what you write actually  sounds good and communicates well what you've done. Yeah, just write down also what you do and why you're doing even if

01:08:45
just to record it because at the end after two months you won't remember why you selected that value for that hyperparameters or you... I mean it could be a comment on the code, it could be some text that you prepare and you keep there on the decisions. Also you have the problem of the synchronization between the team members because there also you can split the work however you want. I don't really care how you split the work among you.

01:09:15
as long as obviously you try to keep things  even. At the end I will ask you what did you do? ah split the workload evenly, but how you split it, it's  up to you. So I don't care exactly who is doing what.  But yeah, try to keep record of what everyone is changing, why, so that you should anyway.

01:09:43
be aware of what the others are doing. I don't want to get at the end where I ask  one of the Oliver's,  what is this part and why you did this way and you reply, that's not my part.  I don't know how that works. In the end, you all put your name in that project. So even if you are not the most expert in  that specific part, I still expect that you know what

01:10:12
what's happening in that section. So you all should have at least a general understanding of where things are going. Please don't get at the end and tell me, ah, that's not my part, because I will definitely go crazy. Okay. And we wouldn't want that. No, it's really bad. All right. And then last part, just about how we collaborate.

01:10:41
I think at least that we should agree on some kind of meeting schedule.  I don't know if it should be having a weekly meeting or bi-weekly meeting.  That's  for my part at least mostly up to you. eh I think meeting more than once  a week would probably be a bit too much for me and Francesco. uh We could have either a bi-weekly meeting  or if you prefer I...

01:11:10
other courses that I did, we had like the students wrote a report, like a weekly report, where I report I mean, we did this, we read this, and we have done this, we started working in this direction, just to keep track of stuff. And when there was the need, they were asking for for for a meeting. So, or we could, we could have it by weekly. And just anyway, have a call.

01:11:40
That's also an option. think weekly is just too frequent.  If you have problems, you are always free to write us. So you don't have to wait two weeks to get feedback on something. But if it is not urgent, bi-weekly for me would be probably better. But again,  you can even think about it and write us later. I'm not asking you to decide right now.

01:12:10
Just tell me how often we can figure it out even in the next days.  If you just figure out how the collaboration works between you, how you write down stuff, how do you report things and then let us know if it is better bi-weekly. ah Weekly could become weekly maybe later when we have, we get closer to the deadline and right now it's just bi-weekly.

01:12:40
And again, yeah, let's figure it out. But I would say I would prefer bi-weekly with the possibility always of you writing us and asking for a meeting and we schedule it much earlier. Also, because if one week you are busy, just write us.  We were busy on other stuff. We couldn't go on with the project. We don't want to do a meeting where you just say, okay, we didn't have time. Waste of time for everyone. Okay.

01:13:07
Yeah, I realize everyone has other stuff to do. Okay. So let's figure it out. Figure out how you want to split the workload and the communications, uh mails if possible, just one of you takes the task to always write the email  to us. So we don't have like four different people writing us emails.  So let's try to get one of the olivers or someone else just to

01:13:37
write us always the email. Always put  in CC Emil or the opposite me CC. uh Yeah. Yes, please create some GitHub  repository and you go for the huge monorepo uh approach or some organization and you just invite us to do that. Your call again,  how to handle that part. uh

01:14:07
Any questions here? Because this I guess this wraps it up. Now is the time for last questions like how the project, like how these maps do exactly what you need to do for this ERS virus. I don't know. It's called Coarse. That's a bit more vague to me. I I received some plans, like you have for sure some, some requirements, but yeah, they are a bit vague.

01:14:36
my mind so please

01:14:41
Any point.

01:14:45
I don't think I have any questions.  I think we'll have to look into it a little bit more  before we get any questions.  Okay, no, no, that makes sense.  I just want to also while you write,  look into the background and other stuff. Now, just also take a look at the course plan because we don't want to make you miss any deadline like halfway we need to

01:15:13
report or the supervisor has to be somewhere, just  let me know because otherwise I may miss it and I don't want to let's say cause problems for your course. So I think that research is an interesting and added part as long as you are with this is hard, this topic is hard.  So you need to put the willpower to go on and the course is just okay, well you pass it somehow.

01:15:42
that's  the two things are  separated. We are more interested in if you like this project, there is the opportunity to  actually push in the machine learning field. But if you feel it's too much in the end, you don't care. ah We will convert to something acceptable for the course. But yeah, we can fine tune the expectation depending on what

01:16:08
what you  think at the  end or you find out that you really hate deep learning, which is even also a possibility,  then in that case you all fail  the exam. Just kidding.  You fail anyway.  Okay. But aside from jokes aside, yeah, let us know if there are some constraints regarding the course because we will try to at least  fix them  in time. yeah, in time.

01:16:37
There's a presentation somewhere in the middle of the court, like four five weeks. I'm not sure if we are needing to be there, but it might be good to least aim to have something to show at that point. Yeah, we want to be obviously the best. Everyone will be super amazed by how good you are. So we will try a presentation where no one is understanding what you are actually doing.

01:17:04
because it's purely planning. we can put a map there, I'm  sure.  Okay.  But  yeah, so please let us know  in advance. So not the day before,  two days before it's already critical, but three days at least would be good. And if you have to produce any presentation that goes to other people, so to your, uh I guess, the course supervisor, don't remember who's that.

01:17:34
just send us first maybe the presentation so we can also have take a look and give you some feedback. So this is part of our job as well. Okay, further questions  on the course. How does it map to your timeline?  You can figure it out also later and write us  but if you have it now, please go ahead.  I have a fast question.  We have  the course.

01:18:03
for what is called Abishek, the one you  remember. And then we have you guys as supervisors, who will be there in the exam?  I think Abishek for sure and then also the supervisors  or just the supervisor because I don't want to get a meal here just for an exam, quite frankly.

01:18:24
I hope only Abishek and I don't have to spend time there.  But no, I'm kidding.  Royal exam with the supervisor as examiner in this intro presentation that I've seen.  so maybe have to go there. Yeah, we'll see. We'll see. Anyway, that's not uh a problem for you. Because in the end, we will produce something that is acceptable for everyone.  And even if we are not there physically.

01:18:53
We should prepare a presentation that shows your work. Yeah, that's right. yeah. Thank you. Maybe a cricket edition and I'm not 100 % sure about this, but I think that like you're doing this on, was it your second semester of the master's what? Okay. Yeah, not the third one. Because I saw the thought that this

01:19:22
course was meant to lead straight into the master's project afterwards so that it was fairly common to,  if you at least like what you are doing in this course, that you'd keep working on that  during your master's project. Have you heard anything about that?  have I just completely misunderstood  that part? No,  I haven't heard anything about that.  But it's also part...

01:19:51
possible to extend the master's thesis. it is possible for us to start it in the next semester. If we want to do that with 10 ECTS in the next. It's something you can think about when you started on the project and if like maybe we get a little part of the way, but there's still a lot of exciting stuff to do then.

01:20:20
Feel free to start thinking about that and discussing it with us at least. I'm sure we can figure something out. Yeah, okay. If there are no more fast, fast, fast questions, I... yeah, that one. Is it possible to get the slides? Oh yeah, sure thing. For me at least. Yeah, sure, sure, sure. Sounds good.

01:20:50
Okay, then we managed to also only go half an hour at times.  Okay,  so actions on your side, let us know how often you want to bi weekly, weekly with  a if you prefer to do bi weekly with  the opportunity of just writing us that that's always there anyway, just write us and we'll see  if you want to  write me up.

01:21:18
bit in advance, also you can stop by my office, but I prefer also if Emil is always in the loop. let's try to keep it, say, with a meeting. Other actions from your side are to understand how this, what we have to do for the course so that we don't miss any deadline for that, because that's also a big part of it. And from our side, we will try to understand

01:21:47
how to give you some computing power, A Raspberry Pi probably. No, I'm again. So yeah, that's it.

01:21:59
Okay, if there are more questions, see you next time. Yeah. Thanks for the meeting. See you around.

01:22:11
Well, I'm thinking about what to do. It sounds pretty hardcore. sounds pretty hardcore. We're pretty easy. But there are a of things we don't know about. What? We're pretty stupid. But there are a of things we don't know about.

01:22:36
But I can also say that if it gets fucked, then they are also okay. Just do something. Just do something, and then it works. I don't you can fix it. He can't judge us. So, if you project, then write him off. I have written him off a research project. Where he hasn't made mistake. That's I'm starting to say. You huge research project. How did you it?

01:23:36
I'll be able in the GitHub repo, so all deadlines are just a little bit. And then we'll a little GitHub link. Would you like to an overview of the time? I don't have 20 minutes, but I'll talk to you later. But can also write it in the repo, then you do it overview. But then you can't do it all at

01:24:01
All  right,

01:24:21
Okay.

01:24:43
My eyes are in a good feedback.

01:24:55
He mentioned can control some gibs and how to refer all the moves.

