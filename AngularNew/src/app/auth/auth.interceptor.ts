import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '../Service/auth.service';

export const tokenInterceptor: HttpInterceptorFn = (request, next) => {
  const authService = inject(AuthService);

  const isRequestAuthorized = authService.isAuthenticated && request.url.startsWith("https://localhost:7223/api/");

  if (isRequestAuthorized) {    
    const token = authService.getToken(); 

    if (token) {
      const clonedRequest = request.clone({
        setHeaders: {
          Authorization: `Bearer ${token}`,
        },
      });

    return next(clonedRequest);
  }
}

  return next(request);
};
