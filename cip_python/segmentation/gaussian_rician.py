import numpy as np
import scipy.special
def compute_gauss_noncentered_rician_negloglikelihood(x,intensity,d):

  alpha=x[0]
  beta=x[1]
  gamma=x[2]
  kappa=x[3]
  mu=x[4]
  sigma=x[5]

  #Set Boundary Conditions
  eps = 0.001
  if kappa < eps:
    kappa=eps
  if sigma < eps:
    sigma=eps
  if mu < eps:
    mu=eps
  
  #Allow negative slope for Gaussian sigma
  if gamma != 0:
    dzero = -kappa/gamma
    if ((dzero > 0) and (dzero <max(d))):
        gamma = (eps-kappa)/max(d)
	
  ## Intensity part of the neglog likelihood (Gaussian)
  ll = (intensity-(alpha*d+beta))**2/(2*(gamma*d+kappa)**2)
	
  ## Distance part of the neglog likelihood (Rician)
  #Check if we are in Gaussian regimen for d
  #Use a plain gaussian to avoid problems with the modified Bessel function
  if mu/sigma**2 > 5:
    ll = ll+np.log(np.sqrt(2*np.pi))+np.log(sigma)+(d-mu)**2/(2*sigma**2)
    
    neglogintegral = np.log(np.sqrt(2*np.pi)) + np.log((kappa+gamma*mu))
        
  else:
    ll = ll-np.log(d)+np.log(sigma**2)+(d**2+mu**2)/(2*sigma**2)-np.log(scipy.special.iv(0,d*mu/sigma**2))
        
    integral=np.sqrt(2*np.pi)/(4*sigma)*np.exp(-mu**2/(4*sigma**2)) * \
        (4*np.exp(mu**2/(4*sigma**2))*kappa/np.sqrt(1/sigma**2) + \
         gamma*np.sqrt(2*np.pi)*(mu**2+2*sigma**2)*scipy.special.iv(0,mu**2/(4*sigma**2)) + \
         gamma*mu**2*np.sqrt(2*np.pi)*scipy.special.iv(1,mu**2/(4*sigma**2)))

    neglogintegral = np.log(integral)


  return sum(ll+neglogintegral)


def compute_gauss_centered_rician_negloglikelihood(x,intensity,d):

  alpha=x[0]
  beta=x[1]
  gamma=x[2]
  kappa=x[3]
  mu=x[4]
  sigma=x[5]
  
  #Set Boundary Conditions
  eps = 0.001
  if kappa < eps:
    kappa=eps
  if sigma < 1.0:
    sigma=eps
  if mu < 1.0:
    mu=eps

  d[d<eps]=eps
    
  #Allow negative slope for Gaussian sigma
  if gamma != 0:
    dzero = (gamma*mu-kappa)/gamma
    if (dzero > 0) & (dzero <max(d)):
        gamma = (eps+gamma*mu-kappa)/max(d)


  ## Intensity part of the neglog likelihood (Gaussian)
  ll = (intensity-(alpha*d+beta))**2/(2*(gamma*(d-mu)+kappa)**2)

  ## Distance part of the neglog likelihood (Rician)
  #Check if we are in Gaussian regimen for d
  #Use a plain gaussian to avoid problems with the modified Bessel function
  if mu/sigma**2 > 5:
    ll = ll+np.log(np.sqrt(2*np.pi))+np.log(sigma)+(d-mu)**2/(2*sigma**2)
    neglogintegral = np.log(np.sqrt(2*np.pi)) + np.log(kappa)
  else:
    ll = ll-np.log(d)+np.log(sigma**2)+(d**2+mu**2)/(2*sigma**2)-np.log(scipy.special.iv(0,d*mu/sigma**2))
    
    integral = np.sqrt(np.pi/2)/(2*sigma) *(4*(kappa-gamma*mu)*sigma+ \
              np.exp(-mu**2/(4*sigma**2))*gamma*np.sqrt(2*np.pi)* \
              ((mu**2+2*sigma**2)*scipy.special.iv(0,mu**2/(4*sigma**2)) + mu**2*scipy.special.iv(1,mu**2/(4*sigma**2))))

    neglogintegral = np.log(integral)

  return sum(ll+neglogintegral)



def gauss_noncentered_rician_pdf(intensity,d,x):


  alpha=x[0]
  beta=x[1]
  gamma=x[2]
  kappa=x[3]
  mu=x[4]
  sigma=x[5]


  pdf=np.zeros(intensity.shape)

  #Mask where we should compute pdf, otherwise is worthless because it is almost zero
  
  mask = ((d> (mu -5*sigma)) & (d < (mu+5*sigma)))
  #print(np.max(d))
  #print(mask)
  #print(np.shape(mask))
  #print(np.shape(intensity))
  #print(np.shape(intensity[mask]))
  #print(np.shape(d[mask]))
  ## Intensity part of the neglog likelihood (Gaussian)
  pdf[mask] = np.exp(-(intensity[mask]-(alpha*d[mask]+beta))**2/(2*(gamma*d[mask]+kappa)**2))


  ## Distance part of the neglog likelihood (Rician)
  #Check if we are in Gaussian regimen for d
  #Use a plain gaussian to avoid problems with the modified Bessel function
  if mu/sigma**2 > 5:
    pdf[mask] = pdf[mask]* 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(d[mask]-mu)**2/(2*sigma**2))
        
    integral = np.sqrt(2*np.pi)*(kappa+gamma*mu)
          
  else:
    pdf[mask] = pdf[mask] * d[mask]/(sigma**2)*np.exp(-(d[mask]**2+mu**2)/(2*sigma**2))*scipy.special.iv(0,d[mask]*mu/sigma**2)
        
    integral=np.sqrt(2*np.pi)/(4*sigma)*np.exp(-mu**2/(4*sigma**2)) * \
          (4*np.exp(mu**2/(4*sigma**2))*kappa/np.sqrt(1/sigma**2) + \
           gamma*np.sqrt(2*np.pi)*(mu**2+2*sigma**2)*scipy.special.iv(0,mu**2/(4*sigma**2)) + \
           gamma*mu**2*np.sqrt(2*np.pi)*scipy.special.iv(1,mu**2/(4*sigma**2)))

  return pdf/integral
  


def gauss_centered_rician_pdf(intensity,d,x):


  alpha=x[0]
  beta=x[1]
  gamma=x[2]
  kappa=x[3]
  mu=x[4]
  sigma=x[5]


  pdf=np.zeros(intensity.shape)

  #Mask where we should compute pdf, otherwise is worthless because it is almost zero
  mask = ((d> (mu -5*sigma)) & (d < (mu+5*sigma)))


  ## Intensity part of the neglog likelihood (Gaussian)
  pdf[mask] = np.exp(-(intensity[mask]-(alpha*d[mask]+beta))**2/(2*(gamma*(d[mask]-mu)+kappa)**2))


  ## Distance part of the neglog likelihood (Rician)
  #Check if we are in Gaussian regimen for d
  #Use a plain gaussian to avoid problems with the modified Bessel function
  if mu/sigma**2 > 5:
    pdf[mask] = pdf[mask]* 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(d[mask]-mu)**2/(2*sigma**2))
    
    integral = np.sqrt(2*np.pi)*kappa

  else:
    pdf[mask] = pdf[mask] * d[mask]/(sigma**2)*np.exp(-(d[mask]**2+mu**2)/(2*sigma**2))*scipy.special.iv(0,d[mask]*mu/sigma**2)
    
    integral = np.sqrt(np.pi/2)/(2*sigma) *(4*(kappa-gamma*mu)*sigma+ \
                np.exp(-mu**2/(4*sigma**2))*gamma*np.sqrt(2*np.pi)* \
                ((mu**2+2*sigma**2)*scipy.special.iv(0,mu**2/(4*sigma**2)) + mu**2*scipy.special.iv(1,mu**2/(4*sigma**2))))

  return pdf/integral


