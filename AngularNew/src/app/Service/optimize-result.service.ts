import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { OptimizeRequest } from '../Models/OptimizeRequest.model';
import { Observable } from 'rxjs';
import { Room } from '../Models/Room.model';

@Injectable({
  providedIn: 'root'
})
export class OptimizeResultService {

  constructor(private _httpClient :HttpClient) { }
  OptimizeCalculation(optimizeRequest: OptimizeRequest):Observable<Room>{
    return this._httpClient.post<Room>(
      'https://localhost:7223/api/OptimizeCalculation',optimizeRequest
    );
  } 
}
