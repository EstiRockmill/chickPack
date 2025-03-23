import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Node } from '../Models/Node.model';


@Injectable({
  providedIn: 'root'
})
export class NodeService {

 constructor(private _httpClient:HttpClient) { }
 getFromServer(): Observable<Node[]> {
     return this._httpClient.get<Node[]>(
     `https://localhost:7223/api/Node`    );
   }
  getByIdFromServer(id: number): Observable<Node> {
    return this._httpClient.get<Node>(
      `https://localhost:7223/api/Node${id}`
    );
  }

  putServer(id: number, n: Node): Observable<any>{
    return this._httpClient.put<any>(`https://localhost:7223/api/Node${id}`, n);
  }

  deleteServer(id: number) :Observable<any>{
    return this._httpClient.delete<any>(`https://localhost:7223/api/Node${id}`);
  }

  addServer(i: Node) :Observable<any>{
   return this._httpClient.post<any>(`https://localhost:7223/api/Node`, i);
  }}
