import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Room } from '../Models/Room.model';
import { RoomPost } from '../Models/RoomPost.model';


@Injectable({
  providedIn: 'root'
})
export class RoomService {

 constructor(private _httpClient:HttpClient) { }
 getFromServer(): Observable<Room[]> {
     return this._httpClient.get<Room[]>(
     `https://localhost:7223/api/Room`    );
   }
  getByIdFromServer(id: number): Observable<Room> {
    return this._httpClient.get<Room>(
      `https://localhost:7223/api/Room${id}`
    );
  }

  putServer(id: number, r: Room): Observable<any>{
    return this._httpClient.put<any>(`https://localhost:7223/api/Room${id}`, r);
  }

  deleteServer(id: number) :Observable<any>{
    return this._httpClient.delete<any>(`https://localhost:7223/api/Room${id}`);
  }

  addServer(r: RoomPost) :Observable<Room>{
   return this._httpClient.post<any>(`https://localhost:7223/api/Room`, r);
  }}
