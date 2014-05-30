PLOT_CDF <- function(degrees) {
  values <- sort(degrees)
  
  elements <- unique(values)
  elements <- rev(elements)
  
  counting <- rep(0,length(elements))
  
  for(i in 1:length(elements))
    counting[i] <- length(values[values==elements[i]])
  
  cdf <- cumsum(1.0*counting)/sum(counting)
  
  plot(rev(elements),rev(cdf),log='y')
}



CONSTRUCT_TREE <- function(nodes,edges){
  
  ids = union()
  
}