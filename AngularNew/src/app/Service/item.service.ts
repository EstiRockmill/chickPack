import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Item } from '../Models/Item.model';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class ItemService {

  constructor(private _httpClient:HttpClient) { }
  getFromServer(): Observable<Item[]> {
      return this._httpClient.get<Item[]>(
      `https://localhost:7223/api/Item`    );
    }
  getByIdFromServer(id: number): Observable<Item> {
    return this._httpClient.get<Item>(
      `https://localhost:7223/api/Item${id}`
    );
  }

  putServer(id: number, i: Item): Observable<any>{
    return this._httpClient.put<any>(`https://localhost:7223/api/Item${id}`, i);
  }

  deleteServer(id: number) :Observable<any>{
    return this._httpClient.delete<any>(`https://localhost:7223/api/Item${id}`);
  }

  addServer(i: Item) :Observable<any>{
   return this._httpClient.post<any>(`https://localhost:7223/api/Item`, i);
  }
}
