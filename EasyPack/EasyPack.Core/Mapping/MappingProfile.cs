using AutoMapper;
using EasyPack.Core.DTO_s;
using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Mapping
{
    public class MappingProfile : Profile
    {
        public MappingProfile() { 
            CreateMap<BoxDTO,Box>().ReverseMap();
            CreateMap<CategoryDTO, Category>().ReverseMap();
            CreateMap<ItemDTO, Item>().ReverseMap();
            CreateMap<RoomDTO, Room>().ReverseMap();
            CreateMap<UserDTO, User>().ReverseMap();
        }
    }
}
