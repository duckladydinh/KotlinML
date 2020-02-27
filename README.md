# KotlinML, Kotlin for Machine Learning
[![Nightly](https://github.com/duckladydinh/KotlinML/workflows/CI%20at%20Night/badge.svg)](https://github.com/duckladydinh/kotlinml/actions?query=workflow%3A%22CI+at+Night%22)
[![On Push](https://github.com/duckladydinh/KotlinML/workflows/CI%20on%20Push/badge.svg)](https://github.com/duckladydinh/kotlinml/actions?query=workflow%3A%22CI+on+Push%22)

KotlinML is a new - and probably the first of its kind - framework for Machine Learning in Kotlin with support for automated hyperparameter optimization (Bayesian Optimization), LightGBM and L-BFGS-B (the C/C++ binding is from another project though).

To welcome the great news that Kotlin for Data Science will be officially supported (December 2019), we aim to make something similar to sklearn and scipy for Kotlin. Will this last more than a school project? Or will it evolve into a work of the same scale of its counterpart in Python? Only time can answer, but I will do my best to keep this project alive!

## Status
Waiting. I am waiting for the release of Kotlin 1.4 and most importantly, Kotlin Numpy, since further advancements at this time will increase the refacturing efforts to replace the current Koma.

## Quick Task(s)
1. Searching for a way to publish an artifact on JCenter or Maven Central.
2. Adding more algorithms & documentation.

## Contribution
This project welcomes and needs your support to survive! Please help me make Kotlin for Data Science great again!

## Basic APIs
For LightGBM, there are a few important methods that you need to know before using. They are presented as follows

```kotlin
    fun fit(                      // from class Booster
        params: Map<String, Any>, // hyperparameters input
        data: Matrix<Double>,     // training data
        label: DoubleArray,       // training output      
        rounds: Int               // number of training iterations
    ): Booster                    // returning fitted model
    
    fun cv(                       // from class Booster
        metric: Metric,           // performance metric
        params: Map<String, Any>, // hyperparameters input
        data: Matrix<Double>,     // training data
        label: DoubleArray,       // training output      
        maxiter: Int,             // number of training iterations
        nFolds: Int               // k in k-fold cross validation
    ): DoubleArray                // returning k scores


    /** 
     * Making predictions for multi- or single input or save
     * These are Booster internal methods
     */
    fun predict(
        data: Matrix<Double>       // multi-input as matrix
    ): DoubleArray                 // multi-output    
    fun predict(
        x: DoubleArray             // single vector input
    ): Double                      // single output
    fun save(filePath: String)
    
    // Please always call this function after all predictions
    fun close()
```

For L-BFGS-B, we provide 2 methods for both maximization and minimization as follows

```kotlin
    fun minimize(                   // from class LBFGSBWrapper
        func: DifferentialFunction, // function to optimize
        xZero: DoubleArray,         // initial guess
        bounds: Array<Bound>,       // constraints
        maxIterations: Int = 15000  // number of iterations
    ): Summary                      // just output summary
    
    fun maximize(                   // from class NumericOptimizer
        func: DifferentialFunction, // function to optimize
        xZero: DoubleArray,         // initial guess
        bounds: Array<Bound>,       // constraints
        maxiter: Int = 15000,       // number of iterations
        type: OptimizerType = OptimizerType.L_BFGS_B // keep this
    )
```

For random and Bayesian optimization, they both inherit a common interface, namely Optimizer which is described below. To use them, just instantiate the optimizer object of your choice (please read the code to know its name, since I may change it tomorrow).

```kotlin
interface Optimizer {
    fun argmax(
        func: (Map<String, Any>) -> Double,// function evaluating
                                           // hyperparameters
        xSpace: XSpace,                    // parameter Domain
        maxiter: Int                       // number of iterations
    ): Pair<Map<String, Any>, Double>
}
```

Last, but not least, is Gaussian Process, which can be used independently as a machine learning model by following the below description:

```kotlin
    fun fit(                          // from class GPRegressor
        data: Matrix<Double>,         // training data
        y: Matrix<Double>,            // training output    
        maxiter: Int = 1,             // number of iterations
        kernel: Kernel = RBF(),       // kernel function
        noise: Double = 1e-10,        // y noise
        normalizeY: Boolean = false   // false means mean is 0
    ): GPRegressor                    // trained Gaussian Process
    
    fun predict(                      // internal of GPRegressor
        x: DoubleArray                // vector input
    ): GPPrediction                   // (mean, variance)
```

That's it. They are the basic APIs we provide at this point. If you have a better way to organize them, please make a pull request.
