import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ImageAnalsys } from '../Models/ImageAnalsys.model';

@Injectable({
  providedIn: 'root'
})
export class AnalyzeService {

  constructor(private _httpClient :HttpClient) { }
  imageAnalysis(image: File):Observable<ImageAnalsys[]>{
    const formData = new FormData();
    formData.append('Image1', image, image.name);
    formData.append('Image2', image, image.name);
    return this._httpClient.post<any[]>(
      'https://localhost:7223/analyze',formData
    );
  } 

  
}
