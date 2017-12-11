
setup <- function() {
    # ensures availability of required packages
    packages <- c("MASS", "rpart", "stats")  # specify packages used below here
    #print('checking installed packages')
    diff = setdiff(packages, rownames(installed.packages()))
    if (length(diff) > 0) {
        #print('installing required packages')
        install.packages(diff, repos="http://cran.r-project.org")
    }
}

predict_rodent_class <- function(segmentation_boundaries, segmentation_value, frequency_time, frequency_value, fs) {
    library('MASS')
    library('rpart')
    library('stats')
    x = segmentation_value
    classes = x + 1
    return(classes)
}
