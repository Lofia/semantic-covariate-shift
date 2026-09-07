library(fdrtool) #https://github.com/cran/fdrtool
#library(gsl)
#library(tggd)
pvt.isoMean = function(y, w)
{
  # Input:	y: measured values in a regression setting
  #		w: weights
  # Output: 	vector containing estimated (isotonic) values
  
  n = length(y)
  
  if(n == 1) return(y)
  else{
    ghat = .C("C_isomean",
              as.double(y),
              as.double(w),
              as.integer(n),
              ghat=double(n), PACKAGE="fdrtool")$ghat
    
    return(ghat)
  }
}

MM=function(x,y=c(),N=Inf,setting='exp',alpha=0,epsilon=0,penalty=FALSE,auto_gcm=TRUE,auto_bisection=FALSE,direct_c=TRUE,...){#,alpha,beta){
  n=length(x);m=length(y)
  x=sort(x);y=sort(y)
  if(alpha>0) {auto_bisection=TRUE;N=Inf;alpha=alpha*n^{-2/3}}
  
  if(setting=='exp'){
    compute_theta0=function(x,y=c()){
      x=c(x,y)
      return(1/mean(x))
    }
    compute_theta_bounds=function(x) return(c(0,5/mean(x))) #?
    bound=c(1e-2,1e4)
    compute_p=function(x,theta){
      x=c(x,Inf)
      return(diff(pexp(x,theta)))
    }
    compute_dp=function(x,theta){ #?
      dF=c(x*exp(-theta*x),0)
      return(dF[2:(n+1)]-dF[1:n])
    }
    dl_dtheta=function(x,theta,v){ # partial derivative: d(l)/d(theta) (ignoring constants, increasing)
      p=compute_p(x,theta)
      dp=compute_dp(x,theta)
      return((N-n)/(1-sum(p*v))*sum(dp*v)-n/theta+sum(x)) ###### changeable
    }
    compute_llf=function(x,y,theta,v){
      xy=c(x,y)
      p=compute_p(x,theta)
      theta0=compute_theta0(x,y)
      p0=compute_p(x,theta0)
      k=epsilon*(1-sum(p))+sum(p*v) #sum(p[1:(n-1)]*v[1:(n-1)])-p[n]
      k0=epsilon*(1-sum(p0))+sum(p0)
      # cat('p',p,'\n','p0',p0,'\n','theta0',theta0,'\n','theta',theta,'\n','v',v)
      if(alpha>0) return(c(llf_all=sum(log(v))+(theta-theta0)*sum(-xy)+(n+m)*log(theta/theta0)-n*log(k/k0)-alpha*n*(1/k-1/k0),
                           llf_v=sum(log(v)),
                           llf_theta=(theta-theta0)*sum(-xy)+n*log(theta/theta0),
                           llf_k=-n*log(k/k0),
                           llf_penalty=-alpha*n*(1/k-1/k0))) # unknown N case
      llf_all=sum(log(v*N/n))+(theta-theta0)*sum(-x)+n*log(theta/theta0)+(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      llf_1=sum(log(v*N/n))
      llf_1_1=sum(log(v*N/n))-n*log(N*(1-c$c)/(N-n))
      llf_1_2=n*log(N*(1-c$c)/(N-n))
      llf_2=(theta-theta0)*sum(-x)
      llf_3=n*log(theta/theta0)
      llf_4=(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      return(c(llf_all=llf_all,llf_1=llf_1,llf_1_1=llf_1_1,llf_1_2=llf_1_2,llf_2=llf_2,llf_3=llf_3,llf_4=llf_4)) ###### changeable
    }
  }
  else if(setting=='half_norm'){
    compute_theta0=function(x) return(-1/2/mean(x^2)) # MLE under null
    compute_theta_bounds=function(x) return(c(-5/2/mean(x^2),0))
    compute_p=function(x,theta){
      x=c(x,Inf)
      sigma=sqrt(-0.5/theta)
      Fx=pnorm(x,sd=sigma)*2-1
      return(diff(Fx))
    }
    compute_dp=function(x,theta){
      dF=-2*x*dnorm(sqrt(-2*theta)*x)/sqrt(-2*theta)
      dF=c(dF,0)
      return(dF[2:(n+1)]-dF[1:n])
      # library(numDeriv)
      # f = function(theta) 2*pnorm(sqrt(-2*theta)*x[1])-1
      # grad(f,-1)
    }
    dl_dtheta=function(x,theta,v){ # partial derivative: d(l)/d(theta) (ignoring constants, increasing)
      p=compute_p(x,theta)
      dp=compute_dp(x,theta)
      return((N-n)/(1-sum(p*v))*sum(dp*v)-n/theta/2-sum(x^2)) ###### changeable
    }
    compute_llf=function(x,theta,v){
      p=compute_p(x,theta)
      theta0=compute_theta0(x)
      p0=compute_p(x,theta0)
      llf_all=sum(log(v*N/n))+(theta-theta0)*sum(x^2)+n/2*log(theta/theta0)+(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      llf_1=sum(log(v*N/n))
      # llf_1_1=sum(log(v*N/n))-n*log(N*(1-c$c)/(N-n))
      # llf_1_2=n*log(N*(1-c$c)/(N-n))
      llf_2=(theta-theta0)*sum(x^2)
      llf_3=n/2*log(theta/theta0)
      llf_4=(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      return(c(llf_all=llf_all,llf_1=llf_1,llf_2=llf_2,llf_3=llf_3,llf_4=llf_4)) ###### changeable llf_1_1=llf_1_1,llf_1_2=llf_1_2,
    }
  }
  else if(setting=='norm'){
    auto_bisection=TRUE
    compute_theta0=function(x,y=c()){
      x=c(x,y)
      mu=mean(x)
      sigma2=var(x)*(n-1)/n
      return(c(-0.5/sigma2,mu/sigma2)) # MLE under null
    }
    # L_bound=c(-Inf,-Inf);R_bound=c(0,Inf)
    compute_p=function(x,theta){
      x=c(x,Inf)
      mu=-theta[2]/2/theta[1]
      sigma=sqrt(-0.5/theta[1])
      Fx=pnorm(x,mean=mu,sd=sigma)
      return(diff(Fx))
    }
    compute_llf=function(x,y,theta,v){
      xy=c(x,y)
      p=compute_p(x,theta)
      theta0=compute_theta0(x,y)
      p0=compute_p(x,theta0)
      k=epsilon*(1-sum(p))+sum(p*v) #sum(p[1:(n-1)]*v[1:(n-1)])-p[n]
      k0=epsilon*(1-sum(p0))+sum(p0)
      # cat('p',p,'\n','p0',p0,'\n','theta0',theta0,'\n','theta',theta,'\n','v',v)
      if(alpha>0) return(c(llf_all=sum(log(v))+(theta[1]-theta0[1])*sum(xy^2)+(theta[2]-theta0[2])*sum(xy)+(n+m)/4*(theta[2]^2/theta[1]-theta0[2]^2/theta0[1])+(n+m)/2*log(theta[1]/theta0[1])-n*log(k/k0)-alpha*n*(1/k-1/k0),
                           llf_v=sum(log(v)),
                           llf_theta=(theta[1]-theta0[1])*sum(xy^2)+(theta[2]-theta0[2])*sum(xy)+(n+m)/4*(theta[2]^2/theta[1]-theta0[2]^2/theta0[1])+(n+m)/2*log(theta[1]/theta0[1]),
                           llf_k=-n*log(k/k0),
                           llf_penalty=-alpha*n*(1/k-1/k0))) # unknown N case
      llf_all=sum(log(v*N/n))+(theta[1]-theta0[1])*sum(x^2)+(theta[2]-theta0[2])*sum(x)+n/4*(theta[2]^2/theta[1]-theta0[2]^2/theta0[1])+n/2*log(theta[1]/theta0[1])+(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      llf_1=sum(log(v*N/n))
      llf_2=(theta[1]-theta0[1])*sum(x^2)+(theta[2]-theta0[2])*sum(x)
      llf_3=n/4*(theta[2]^2/theta[1]-theta0[2]^2/theta0[1])+n/2*log(theta[1]/theta0[1])
      llf_4=(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      return(c(llf_all=llf_all,llf_1=llf_1,llf_2=llf_2,llf_3=llf_3,llf_4=llf_4))
    }
  }
  else if(setting=='gamma'){
    auto_bisection=TRUE
    compute_theta0=function(x,y=c()){
      x=c(x,y)
      beta=mean(x)/var(x)*n/(n-1)
      alpha=beta*mean(x)
      return(c(alpha,beta)) # MLE under null
    }
    # L_bound=c(-Inf,-Inf);R_bound=c(0,Inf)
    compute_p=function(x,theta){
      x=c(x,Inf)
      Fx=pgamma(x,theta[1],theta[2])
      return(diff(Fx))
    }
    compute_llf=function(x,y,theta,v){
      xy=c(x,y)
      # if(theta[1]<=1 | theta[2]<=0) return(llf_all=-Inf)
      if(sum(theta<1e-9)) theta[theta<1e-9]=1e-9
      p=compute_p(x,theta)
      theta0=compute_theta0(x,y)
      p0=compute_p(x,theta0)
      k=epsilon*(1-sum(p))+sum(p*v) #sum(p[1:(n-1)]*v[1:(n-1)])-p[n]
      k0=epsilon*(1-sum(p0))+sum(p0)
      # cat('p',p,'\n','p0',p0,'\n','theta0',theta0,'\n','theta',theta,'\n','v',v)
      if(alpha>0) return(llf_all=sum(log(v))+(theta[1]-theta0[1])*sum(log(xy))+(theta[2]-theta0[2])*sum(-xy)+(n+m)*(theta[1]*log(theta[2])-log(gamma(theta[1]))-theta0[1]*log(theta0[2])+log(gamma(theta0[1])))-n*log(k/k0)-alpha*n*(1/k-1/k0)) # unknown N case
      llf_all=sum(log(v*N/n))+(theta[1]-theta0[1])*sum(log(x))+(theta[2]-theta0[2])*sum(-x)+n*(theta[1]*log(theta[2])-log(gamma(theta[1]))-theta0[1]*log(theta0[2])+log(gamma(theta0[1])))+(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      llf_1=sum(log(v*N/n))
      llf_2=(theta[1]-theta0[1])*sum(log(x))+(theta[2]-theta0[2])*sum(-x)
      llf_3=n*(theta[1]*log(theta[2])-log(gamma(theta[1]))-theta0[1]*log(theta0[2])+log(gamma(theta0[1])))
      llf_4=(N-n)*log((1-sum(p*v))/(1-sum(p0*n/N)))
      return(c(llf_all=llf_all,llf_1=llf_1,llf_2=llf_2,llf_3=llf_3,llf_4=llf_4))
    }
  }
  else if(setting=='tggd'){ # truncated generalized gamma distribution
    q=list(...)$xmin
    if(min(x)<=q) stop('ERROR: x<xmin')
    auto_bisection=TRUE
    compute_theta0=function(x,y=c()){
      x=c(x,y)
      beta=mean(x)/var(x)
      alpha=beta*mean(x)
      return(c(alpha,beta)) # simple guess under null, theta1=alpha=a+1, theta2=beta=lambda=b
    }
    compute_p=function(x,theta){
      # x=c(x,Inf)
      Fx=(gsl::gamma_inc(theta[1],theta[2]*q)-gsl::gamma_inc(theta[1],theta[2]*x))/gsl::gamma_inc(theta[1],theta[2]*q)*(x>q) #F(x)
      Fx=c(Fx,1) # gsl::gamma_inc does not allow x=Inf
      return(diff(Fx))
    }
    compute_llf=function(x,y,theta,v){
      xy=c(x,y)
      if(sum(theta<1e-9)) theta[theta<1e-9]=1e-9
      p=compute_p(x,theta)
      theta0=compute_theta0(x,y)
      p0=compute_p(x,theta0)
      k=epsilon*(1-sum(p))+sum(p*v) #sum(p[1:(n-1)]*v[1:(n-1)])-p[n]
      k0=epsilon*(1-sum(p0))+sum(p0)
      # cat('p',p,'\n','p0',p0,'\n','theta0',theta0,'\n','theta',theta,'\n','v',v)
      if(alpha>0) return(llf_all=sum(log(v))+(theta[1]-theta0[1])*sum(log(xy))+(theta[2]-theta0[2])*sum(-xy)+(n+m)*(theta[1]*log(theta[2])-log(gsl::gamma_inc(theta[1],theta[2]*q))-theta0[1]*log(theta0[2])+log(gsl::gamma_inc(theta0[1],theta0[2]*q)))-n*log(k/k0)-alpha*n*(1/k-1/k0)) # very similar to gamma case
    }
  }
  else if(setting=='pois'){
    auto_bisection=TRUE
    compute_theta0=function(x,y=c()){
      x=c(x,y)
      lambda=mean(x)
      return(lambda) # MLE under null
    }
    bound=c(1e-2,1e4)
    compute_p=function(x,theta){
      x=c(x,Inf)
      Fx=ppois(x,lambda=theta)-dpois(x,lambda=theta)
      return(diff(Fx))
    }
    compute_llf=function(x,y,theta,v){
      xy=c(x,y)
      p=compute_p(x,theta)
      theta0=compute_theta0(x,y)
      p0=compute_p(x,theta0)
      k=epsilon*(1-sum(p))+sum(p*v) #sum(p[1:(n-1)]*v[1:(n-1)])-p[n]
      k0=epsilon*(1-sum(p0))+sum(p0)
      # cat('p',p,'\n','p0',p0,'\n','theta0',theta0,'\n','theta',theta,'\n','v',v)
      if(alpha>0) return(c(llf_all=sum(log(v))+(log(theta[1])-log(theta0[1]))*sum(xy)-(n+m)*(theta-theta0)-n*log(k/k0)-alpha*n*(1/k-1/k0))) # unknown N case
    }
  }
  else if(setting=='binom'){
    size=list(...)$size
    auto_bisection=TRUE
    compute_theta0=function(x,y=c()){
      x=c(x,y)
      prob=mean(x)/size
      return(prob) # MLE under null
    }
    bound=c(1e-2,1)
    compute_p=function(x,theta){
      x=c(x,Inf)
      Fx=pbinom(x,size=size,prob=theta)-dbinom(x,size=size,prob=theta)
      return(diff(Fx))
    }
    compute_llf=function(x,y,theta,v){
      xy=c(x,y)
      p=compute_p(x,theta)
      theta0=compute_theta0(x,y)
      p0=compute_p(x,theta0)
      k=epsilon*(1-sum(p))+sum(p*v) #sum(p[1:(n-1)]*v[1:(n-1)])-p[n]
      k0=epsilon*(1-sum(p0))+sum(p0)
      # cat('p',p,'\n','p0',p0,'\n','theta0',theta0,'\n','theta',theta,'\n','v',v)
      if(alpha>0) return(c(llf_all=sum(log(v))+(log(theta[1])-log(theta0[1]))*sum(xy)+(log(1-theta[1])-log(1-theta0[1]))*sum(size-xy)-n*log(k/k0)-alpha*n*(1/k-1/k0))) # unknown N case
    }
  }
  
  ####################################
  ####################################
  
  # pava=function(dx,dy){
  #   n=length(dx)
  #   slopes=dy/dx
  #   index=1:n
  #   j=n
  #   for(j in (n-1):1){
  #     while(slopes[j+1]<slopes[j]){
  #       dx[j]=dx[j+1]+dx[j];dx=dx[-(j+1)]
  #       dy[j]=dy[j+1]+dy[j];dy=dy[-(j+1)]
  #       slopes[(j)]=dy[j]/dx[j];slopes=slopes[-(j+1)]
  #       index=index[-(j+1)]
  #       if(length(index)==j) break
  #     }
  #   }
  #   return(rep(slopes,diff(c(index,n+1))))
  # }
  # pava_gcmlcm=function(dx,dy){
  #   u=gcmlcm(cumsum(c(0,dx)),cumsum(c(0,dy)))
  #   index=diff(match(u$x.knots,cumsum(c(0,dx)))-1)
  #   return(rep(u$slope.knots,index))
  # }
  # t1=Sys.time()
  # for(i in 1:1000) a=pava_gcmlcm(runif(1000),runif(1000))
  # t2=Sys.time()
  # t2-t1
  #
  # t1=Sys.time()
  # for(i in 1:1000) pava(runif(10000),runif(10000))
  # t2=Sys.time()
  # t2-t1
  #
  # t1=Sys.time()
  # for(i in 1:1000) dx=runif(1000);dy=runif(1000);a=pvt.isoMean(dy/dx,dx)
  # t2=Sys.time()
  # t2-t1
  
  if(auto_gcm && direct_c){
    # library(fdrtool)
    est=function(c,theta){ #careful with ties
      a=rep(1,n)#;a[1]=a[1]+n*alpha
      b=(N-n)/(1-c)*p;  b[n]=b[n]+penalty*(N-n)/(1-c)*2/sqrt(n) # penalty added
      if(alpha>0) b=(c-alpha)/c^2*n*p
      
      position=c(0,which(b>0))
      # if(tail(position,1)<n) position=c(position,n)
      aa=diff(position) # a without ties #debugged on 12/25/2024 diff(c(which(b>0),n+1))
      bb=b[b>0] # b without ties
      # u=gcmlcm(cumsum(c(0,bb)),cumsum(c(0,aa)))
      # index=diff(match(u$x.knots,cumsum(c(0,bb)))-1)
      # vv=rep(u$slope.knots,index)
      vv=pvt.isoMean(aa/bb,bb)
      v=rep(vv,aa)
      v=sapply(v,min,1)
      if(alpha>0) v=sapply(v,max,epsilon)
      return(list(v=v))#,v_tilde=u$slope.knots,index=index)) ###### shortenable
    }
  }
  else if(auto_gcm && !direct_c){
    # library(fdrtool)
    est=function(c,theta){ #careful with ties
      a=rep(1,n)#;a[1]=a[1]+n*alpha
      b=(N-n)/(1-c)*p;  b[n]=b[n]+penalty*(N-n)/(1-c)*2/sqrt(n) # penalty added
      if(alpha>0) b=(c-alpha)/c^2*n*p
      
      aa=diff(c(which(b>0),n+1)) # a without ties
      bb=b[b>0] # b without ties
      u=gcmlcm(cumsum(c(0,bb)),cumsum(c(0,aa)))
      index=diff(match(u$x.knots,cumsum(c(0,bb)))-1)
      vv=rep(u$slope.knots,index)
      # vv=pvt.isoMean(aa/bb,bb)
      v=rep(vv,aa)
      v=sapply(v,min,1)
      if(alpha>0) v=sapply(v,max,epsilon)
      return(list(v=v))#,v_tilde=u$slope.knots,index=index)) ###### shortenable
    }
  }
  else{
    est=function(c,theta){
      a=rep(1,n)#;a[1]=a[1]+n*alpha
      b=(N-n)/(1-c)*p;  b[n]=b[n]+penalty*(N-n)/(1-c)*2/sqrt(n) # penalty added
      if(alpha>0) b=(c-alpha)/c^2*n*p
      
      M=matrix(nrow=n,ncol=n)
      for(i in 1:n) for(j in i:n) M[i,j]=sum(a[i:j])/sum(b[i:j])
      
      v=rep(Inf,n)
      v[1]=min(M[1,]);v[n]=max(M[,n])
      for(k in 2:(n-1)) v[k]=max(apply(M[1:k,k:n],1,min))
      # cat(sum(p*v))
      # cat('\n')
      v=sapply(v,min,1)
      if(alpha>0) v=sapply(v,max,epsilon)
      return(list(v=v,v_tilde=unique(v),index=match(unique(v),v),M=M)) ###### shortenable
    }
  }
  
  bisection_c=function(left,right,theta,iter=1){ # f increasing
    mid=(left+right)/2
    fmid=mid-epsilon*(1-sum(p))-sum(p*est(mid,theta)$v) ###### changeable?
    if(abs(left-right)<0.0001) return(list(c=mid,iter=iter,fmid=fmid))
    iter=iter+1
    # cat(c(left,right,mid,fmid))
    # cat('\n')
    if(fmid>=0) return(bisection_c(left,mid,theta,iter))
    if(fmid<0) return(bisection_c(mid,right,theta,iter))
  }
  
  bisection_theta=function(left,right,v,iter=1){ # f increasing
    mid=(left+right)/2
    fmid=dl_dtheta(x,mid,v)
    if(abs(left-right)<0.00001) return(list(theta=mid,iter=iter,fmid=fmid))
    iter=iter+1
    # cat(c(left,right,mid,fmid))
    # cat('\n')
    if(fmid>=0) return(bisection_theta(left,mid,v,iter))
    if(fmid<0) return(bisection_theta(mid,right,v,iter))
  }
  
  theta0=compute_theta0(x,y)
  #theta0=mean(x)/var(x) # under HA, mean=(b+1)/theta, var=(b+1)/theta^2
  
  ####################################
  ####################################
  
  loop=1
  repeat{
    # theta0=c(0.6,0.006)
    # cat('\n','theta0 =',theta0,'\n')
    p=compute_p(x,theta0)
    # cat('\n','theta0 =',theta0,'\n')
    
    c=bisection_c(alpha+0.001,1,theta0)
    h=0
    while(N==Inf & h<10){
      h=h+1
      c2=bisection_c(c$c,1,theta0)
      if(c$c-c2$c<1e-4) h=999
      c=c2
    }
    if(h>0 & h<999) warning('c iter error')
    
    v=est(c$c,theta0)#;v$v=rep(0.8,n)
    # cat(loop,'\n',
    #     'c:',c$c,c$iter,c$fmid,'\n',
    #     'v:',v$v_tilde,v$index,'\n')
    if(auto_bisection){
      if(length(theta0)>1){
        out=optim(theta0,function(u)
          return(compute_llf(x,y,u,v$v)[1]),control=list(fnscale=-1),gr=NULL)
        # cat('\n out')
        theta=list(theta=out$par,iter=out$value,fmid=out$convergence)
      }
      else if(length(theta0)==1){
        out=optimize(function(u)
          return(compute_llf(x,y,u,v$v)[1]),bound,maximum=TRUE)
        # cat('\n out')
        theta=list(theta=out$maximum,iter=out$objective,fmid=NULL)
      }
      else stop('ERROR_AUTO_BISECTION')
    }
    else{
      bounds=compute_theta_bounds(x)
      theta=bisection_theta(bounds[1],bounds[2],v$v)
    }
    
    llf=compute_llf(x,y,theta$theta,v$v)
    
    # cat(loop,'\n',
    #     'c:',c$c,c$iter,c$fmid,'\n',
    #     'v:',head(v$v,100),'\n',#'(',v$index,')',
    #     'theta:',theta$theta,theta$iter,theta$fmid,'\n',
    #     #'llf0:',llf0,'\n',
    #     'llf:',llf,sum(p*v$v),'\n')
    
    # cat('theta:',theta$theta,theta$iter,theta$fmid,'\n')
    
    if(sum(abs(theta$theta-theta0))<0.001){
      if(setting=='norm') {
        theta=theta$theta
        mu=-theta[2]/2/theta[1]
        sigma=sqrt(-0.5/theta[1])
        return(list(v=v$v,theta=c(mu,sigma),llf=llf,c=c$c,loop=loop)) #,llf_v=llf[[length(llf)]]
      }
      return(list(v=v$v,theta=theta$theta,llf=llf,c=c$c,loop=loop))
    }
    #_tilde,index=v$index
    theta0=theta$theta #;v0=v
    loop=loop+1
  }
}

# u=MM(x$x,x$y,setting='norm',alpha=0.7,epsilon=0.2);mean(x$x);u$theta;u$c